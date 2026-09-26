"""Security and durability tests for the model-independent gateway."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

from nps_gateway.contracts import MAX_PROPOSAL_BYTES, TOOLS, canonical_json, parse_proposal, strict_json
from nps_gateway.runtime import messages_for, run_copy
from nps_gateway.store import Gateway


class FixedClock:
    def __init__(self, value: float = 1_000.0):
        self.value = value

    def __call__(self) -> float:
        return self.value


class Adapter:
    def __init__(self, result=None, error: Exception | None = None):
        self.result = result
        self.error = error
        self.messages = None
        self.tools = None

    def generate(self, messages, tools):
        self.messages, self.tools = messages, tools
        if self.error:
            raise self.error
        return self.result


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "gateway.sqlite"
        self.clock = FixedClock()
        self.gateway = Gateway(self.db_path, clock=self.clock)
        self.source = {"source_id": "approved", "record_id": "record-1", "value": "copy exactly"}
        self.source_raw = canonical_json(self.source).encode()
        self.source_digest = hashlib.sha256(self.source_raw).hexdigest()
        self.gateway.import_record(self.source_raw, expected_sha256=self.source_digest)

    def tearDown(self):
        self.tempdir.cleanup()

    @staticmethod
    def proposal(note_id="note-1", content="copy exactly"):
        return canonical_json({"name": "write_note", "arguments": {"note_id": note_id, "content": content}})

    def test_happy_path_persists_exact_source_provenance_and_audit(self):
        grant = self.gateway.issue("alice", "approved", "record-1", "note-1")
        decision = self.gateway.execute("alice", grant.token, self.proposal())

        self.assertTrue(decision.allowed)
        self.assertEqual("authorized_source_copy", decision.reason)
        self.assertEqual(
            [{"subject": "alice", "note_id": "note-1", "content": "copy exactly",
              "source_id": "approved", "record_id": "record-1", "source_sha256": self.source_digest,
              "version": 1, "run_id": grant.run_id}],
            self.gateway.notes("alice"),
        )
        self.assertEqual("executed", self.gateway.audit("alice")[0]["status"])
        self.assertTrue(self.gateway.audit("alice")[0]["side_effect"])

    def test_restart_and_independent_instances_allow_only_one_replay(self):
        grant = self.gateway.issue("alice", "approved", "record-1", "note-1")
        restarted = Gateway(self.db_path, clock=self.clock)
        self.assertTrue(restarted.execute("alice", grant.token, self.proposal()).allowed)
        again = Gateway(self.db_path, clock=self.clock).execute("alice", grant.token, self.proposal())
        self.assertEqual(("blocked", "replay_or_revoked"), (again.status, again.reason))

    def test_concurrent_independent_instances_spend_token_once(self):
        grant = self.gateway.issue("alice", "approved", "record-1", "note-1")
        barrier = threading.Barrier(3)
        results = []

        def spend():
            instance = Gateway(self.db_path, clock=self.clock)
            barrier.wait()
            results.append(instance.execute("alice", grant.token, self.proposal()))

        workers = [threading.Thread(target=spend) for _ in range(2)]
        for worker in workers:
            worker.start()
        barrier.wait()
        for worker in workers:
            worker.join(timeout=5)
            self.assertFalse(worker.is_alive())
        self.assertEqual(1, sum(result.allowed for result in results))
        self.assertEqual(["replay_or_revoked"], [result.reason for result in results if not result.allowed])

    def test_grant_is_exactly_bound_to_subject_note_and_content(self):
        grant = self.gateway.issue("alice", "approved", "record-1", "note-1")
        wrong_subject = self.gateway.execute("bob", grant.token, self.proposal())
        self.assertEqual("subject_mismatch", wrong_subject.reason)
        self.assertTrue(self.gateway.execute("alice", grant.token, self.proposal()).allowed)

        for proposal, expected in ((self.proposal("other"), "scope_mismatch"),
                                   (self.proposal("note-2", "tampered"), "argument_binding_mismatch")):
            grant = self.gateway.issue("alice", "approved", "record-1", "note-2")
            result = self.gateway.execute("alice", grant.token, proposal)
            self.assertEqual(expected, result.reason)
            self.assertEqual("replay_or_revoked", self.gateway.execute("alice", grant.token, self.proposal("note-2")).reason)

    def test_expiry_and_revocation_are_terminal(self):
        grant = self.gateway.issue("alice", "approved", "record-1", "note-1", ttl_seconds=1)
        self.clock.value += 1
        self.assertEqual("expired", self.gateway.execute("alice", grant.token, self.proposal()).reason)

        grant = self.gateway.issue("alice", "approved", "record-1", "note-2")
        self.assertEqual("host_revoked", self.gateway.revoke(grant).reason)
        self.assertEqual("replay_or_revoked", self.gateway.execute("alice", grant.token, self.proposal("note-2")).reason)

    def test_bad_or_hostile_json_cannot_authorize_and_spends_matching_grant(self):
        invalid = [
            '{"name":"write_note","name":"write_note","arguments":{}}',
            '{"name":"write_note","arguments":{"note_id":"note-1","content":NaN}}',
            '{"name":"write_note","arguments":{"note_id":"note-1","content":"\\ud800"}}',
            '[]',
            '{"name":"write_note","arguments":{"note_id":1,"content":"copy exactly"}}',
            "{" + '"x":' + '"a"' * (MAX_PROPOSAL_BYTES + 1) + "}",
        ]
        for raw in invalid:
            grant = self.gateway.issue("alice", "approved", "record-1", "note-1")
            result = self.gateway.execute("alice", grant.token, raw)
            self.assertFalse(result.allowed, raw[:40])
            self.assertEqual("replay_or_revoked", self.gateway.execute("alice", grant.token, self.proposal()).reason)

        self.assertEqual({"a": 1}, strict_json('{"a":1}'))
        with self.assertRaises(ValueError):
            parse_proposal('{"name":"delete_everything","arguments":{}}')

    def test_messages_and_tools_never_disclose_grant_or_capability(self):
        grant = self.gateway.issue("alice", "approved", "record-1", "note-1")
        messages = messages_for(grant, "untrusted annotation")
        rendered = canonical_json(messages)
        self.assertNotIn(grant.token, rendered)
        self.assertNotIn(grant.run_id, rendered)
        self.assertNotIn("token", rendered)
        self.assertNotIn("capability", canonical_json(TOOLS))

        adapter = Adapter(self.proposal("note-2"))
        self.assertTrue(run_copy(self.gateway, adapter, subject="alice", source_id="approved", record_id="record-1", note_id="note-2").allowed)
        self.assertNotIn("token", canonical_json(adapter.messages))
        self.assertNotIn("capability", canonical_json(adapter.tools))

    def test_provider_monitor_errors_and_veto_never_authorize(self):
        cases = [
            (Adapter(error=RuntimeError("secret provider body")), None, "provider_error"),
            (Adapter(self.proposal()), lambda _proposal: (_ for _ in ()).throw(RuntimeError()), "monitor_error"),
            (Adapter(self.proposal()), lambda _proposal: False, "monitor_veto"),
        ]
        for index, (adapter, veto, reason) in enumerate(cases):
            result = run_copy(self.gateway, adapter, subject="alice", source_id="approved", record_id="record-1",
                              note_id=f"note-{index}", veto=veto)
            self.assertEqual(("error" if reason.endswith("error") else "blocked", reason), (result.status, result.reason))
            self.assertEqual([], self.gateway.notes("alice"))

    def test_stale_destination_and_principal_separation(self):
        first = self.gateway.issue("alice", "approved", "record-1", "note-1")
        stale = self.gateway.issue("alice", "approved", "record-1", "note-1")
        self.assertTrue(self.gateway.execute("alice", first.token, self.proposal()).allowed)
        self.assertEqual("stale_destination", self.gateway.execute("alice", stale.token, self.proposal()).reason)
        bob = self.gateway.issue("bob", "approved", "record-1", "note-1")
        self.assertTrue(self.gateway.execute("bob", bob.token, self.proposal()).allowed)
        self.assertEqual(1, len(self.gateway.notes("alice")))
        self.assertEqual(1, len(self.gateway.notes("bob")))
        self.assertNotEqual(self.gateway.notes("alice")[0]["run_id"], self.gateway.notes("bob")[0]["run_id"])
        self.assertTrue(all(event["subject"] == "alice" for event in self.gateway.audit("alice")))

    def test_sqlite_abort_rolls_back_grant_consumption_note_and_audit(self):
        grant = self.gateway.issue("alice", "approved", "record-1", "note-1")
        with sqlite3.connect(self.db_path) as db:
            db.execute("CREATE TRIGGER abort_note BEFORE INSERT ON notes BEGIN SELECT RAISE(ABORT, 'test abort'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.gateway.execute("alice", grant.token, self.proposal())
        self.assertEqual([], self.gateway.notes("alice"))
        with sqlite3.connect(self.db_path) as db:
            db.execute("DROP TRIGGER abort_note")
        self.assertTrue(self.gateway.execute("alice", grant.token, self.proposal()).allowed)

    def test_source_digest_mismatch_and_immutable_provenance_are_rejected(self):
        altered = canonical_json({**self.source, "value": "different"}).encode()
        with self.assertRaisesRegex(ValueError, "source_digest_mismatch"):
            self.gateway.import_record(altered, expected_sha256=self.source_digest)
        with self.assertRaisesRegex(ValueError, "source_is_immutable"):
            self.gateway.import_record(altered, expected_sha256=hashlib.sha256(altered).hexdigest())


if __name__ == "__main__":
    unittest.main()
