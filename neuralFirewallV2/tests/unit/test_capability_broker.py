"""Security-contract tests for the CPU-only reference capability broker."""

from __future__ import annotations

import json
import threading
import unittest

from neuralFirewallV2.src.policy.capability_broker import (
    CapabilityBroker,
    TrustedRecord,
    canonical_json,
    parse_native_request,
)


SECRET = b"local-test-secret-do-not-use-in-production-32bytes"


def wire(value: str, resource: str = "notes") -> str:
    body = canonical_json(
        {"name": "write_record", "arguments": {"resource": resource, "value": value}}
    )
    return f"<tool_call>\n{body}\n</tool_call><|im_end|>"


class CapabilityBrokerTest(unittest.TestCase):
    def setUp(self):
        self.now = 1000.0
        self.broker = CapabilityBroker(SECRET, clock=lambda: self.now)
        self.subject = "session:test"
        self.workspace = {}
        self.record = TrustedRecord("public-index", "rec-17", "verified fact")

    def test_source_bound_policy_uses_trusted_record_and_blocks_mutation(self):
        token = self.broker.issue_source_bound(self.subject, self.record)
        good = self.broker.authorize_and_execute(
            self.subject, wire("verified fact"), token, self.workspace
        )
        self.assertEqual((good.allowed, good.reason, good.side_effect), (True, "authorized", True))
        self.assertEqual(self.workspace, {"notes": "verified fact"})

        another = self.broker.issue_source_bound(self.subject, self.record)
        denied = self.broker.authorize_and_execute(
            self.subject, wire("attacker substituted value"), another, self.workspace
        )
        self.assertEqual(denied.reason, "argument_binding_mismatch")
        self.assertEqual(self.workspace, {"notes": "verified fact"})
        self.assertEqual(
            self.broker.authorize_and_execute(
                self.subject, wire(self.record.value), another, self.workspace
            ).reason,
            "replay",
        )

    def test_scope_only_is_explicitly_less_restrictive(self):
        token = self.broker.issue_scope(self.subject)
        decision = self.broker.authorize_and_execute(
            self.subject, wire("untrusted but in-scope content"), token, self.workspace
        )
        self.assertTrue(decision.allowed)
        self.assertEqual(self.workspace["notes"], "untrusted but in-scope content")

    def test_known_but_unissued_message_tool_is_valid_protocol_but_denied(self):
        message = (
            '<tool_call>{"name":"send_mock_message","arguments":'
            '{"destination":"outbox","content":"external request"}}</tool_call>'
        )
        parsed = parse_native_request(message)
        self.assertEqual(parsed["tool"], "send_mock_message")
        token = self.broker.issue_scope(self.subject)
        decision = self.broker.authorize_and_execute(
            self.subject, message, token, self.workspace
        )
        self.assertEqual(decision.reason, "scope_mismatch")
        self.assertFalse(decision.side_effect)

    def test_oracle_control_is_separate_and_exact(self):
        args = {"resource": "notes", "value": "known expected answer"}
        token = self.broker.issue_oracle_bound(self.subject, args)
        self.assertTrue(
            self.broker.authorize_and_execute(
                self.subject, wire(args["value"]), token, self.workspace
            ).allowed
        )

    def test_source_issuer_contract_has_no_expected_answer_parameter(self):
        import inspect

        parameters = inspect.signature(self.broker.issue_source_bound).parameters
        self.assertEqual(set(parameters), {"subject", "record" , "ttl_seconds"})

    def test_replay_and_expiry_are_denied(self):
        token = self.broker.issue_source_bound(self.subject, self.record)
        first = self.broker.authorize_and_execute(
            self.subject, wire(self.record.value), token, self.workspace
        )
        second = self.broker.authorize_and_execute(
            self.subject, wire(self.record.value), token, self.workspace
        )
        self.assertTrue(first.allowed)
        self.assertEqual(second.reason, "replay")

        expired = self.broker.issue_scope(self.subject, ttl_seconds=1)
        self.now += 2
        self.assertEqual(
            self.broker.authorize_and_execute(
                self.subject, wire("fact"), expired, self.workspace
            ).reason,
            "expired",
        )

    def test_concurrent_replay_has_at_most_one_effect(self):
        token = self.broker.issue_source_bound(self.subject, self.record)
        barrier = threading.Barrier(8)
        decisions = []
        lock = threading.Lock()

        def attempt():
            barrier.wait()
            result = self.broker.authorize_and_execute(
                self.subject, wire(self.record.value), token, self.workspace
            )
            with lock:
                decisions.append(result)

        workers = [threading.Thread(target=attempt) for _ in range(8)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        self.assertEqual(sum(item.allowed for item in decisions), 1)
        self.assertEqual(sum(item.reason == "replay" for item in decisions), 7)

    def test_denies_missing_tampered_wrong_subject_and_model_claim(self):
        raw = wire(self.record.value)
        self.assertEqual(
            self.broker.authorize_and_execute(self.subject, raw, None, self.workspace).reason,
            "missing_or_untrusted_token",
        )
        token = self.broker.issue_source_bound(self.subject, self.record)
        altered = json.loads(token)
        altered["payload"]["resource"] = "protected"
        self.assertEqual(
            self.broker.authorize_and_execute(
                self.subject, raw, canonical_json(altered), self.workspace
            ).reason,
            "invalid_signature",
        )
        token = self.broker.issue_source_bound(self.subject, self.record)
        self.assertEqual(
            self.broker.authorize_and_execute(
                "session:other", raw, token, self.workspace
            ).reason,
            "subject_mismatch",
        )
        model_claim = (
            '<tool_call>{"name":"write_record","arguments":{"resource":"notes",'
            '"value":"verified fact"},"capability":"admin"}</tool_call>'
        )
        self.assertEqual(
            self.broker.authorize_and_execute(
                self.subject, model_claim, token, self.workspace
            ).reason,
            "wire_schema",
        )

    def test_parser_rejects_duplicate_keys_prose_and_nonfinite_values(self):
        invalid = (
            '<tool_call>{"name":"write_record","name":"shell","arguments":{}}</tool_call>',
            wire("fact") + " trailing",
            '<tool_call>{"name":"write_record","arguments":{"resource":"notes",'
            '"value":NaN}}</tool_call>',
        )
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_native_request(raw)

    def test_constructor_and_input_limits_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "32_bytes"):
            CapabilityBroker(b"short")
        token = self.broker.issue_scope(self.subject)
        oversized = "<tool_call>" + " " * 5000 + "</tool_call>"
        self.assertEqual(
            self.broker.authorize_and_execute(
                self.subject, oversized, token, self.workspace
            ).reason,
            "wire_too_large",
        )


if __name__ == "__main__":
    unittest.main()
