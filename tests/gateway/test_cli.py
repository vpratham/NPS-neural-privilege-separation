"""Exercise the shipped entrypoint and real persistent effects without inference."""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from nps_gateway.cli import main


class CLITests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.db = self.root / "gateway.sqlite3"

    def tearDown(self):
        self.directory.cleanup()

    def invoke(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["--db", str(self.db), *args])
        return code, json.loads(out.getvalue() or err.getvalue())

    def test_offline_demo_and_persistent_read_commands(self):
        code, result = self.invoke("demo")
        self.assertEqual(code, 0)
        self.assertTrue(result["demo_passed"])
        self.assertEqual(result["inference"], "none; deterministic integration demo")
        self.assertEqual(sum(row["side_effect"] for row in result["cases"]), 2)
        self.assertEqual(self.invoke("notes")[1], result["notes"])
        self.assertEqual(self.invoke("audit", "--limit", "1")[1][0]["reason"], "replay_or_revoked")

    def test_import_and_external_model_proposal_execute_or_block(self):
        source = self.root / "source.json"
        raw = b'{"source_id":"handbook","record_id":"v1","value":"approved text"}'
        source.write_bytes(raw)
        code, _ = self.invoke("import-source", str(source), "--sha256", hashlib.sha256(raw).hexdigest())
        self.assertEqual(code, 0)
        proposal = self.root / "proposal.json"
        proposal.write_text(json.dumps({"name": "write_note", "arguments": {"note_id": "test", "content": "approved text"}}))
        args = ("run", "--source", "handbook", "--record", "v1", "--note", "test",
                "--provider", "file", "--proposal-file", str(proposal))
        self.assertEqual(self.invoke(*args)[0], 0)
        proposal.write_text(json.dumps({"name": "write_note", "arguments": {"note_id": "test", "content": "poisoned"}}))
        code, result = self.invoke(*args)
        self.assertEqual((code, result["reason"]), (2, "argument_binding_mismatch"))
        self.assertEqual(self.invoke("notes")[1][0]["content"], "approved text")

    def test_bad_digest_and_corrupt_database_report_errors(self):
        source = self.root / "source.json"
        source.write_text("{}")
        code, result = self.invoke("import-source", str(source), "--sha256", "\u00e9")
        self.assertEqual((code, result["status"]), (1, "error"))
        self.db.write_bytes(b"invalid sqlite database")
        self.assertEqual(self.invoke("notes")[0], 1)


if __name__ == "__main__":
    unittest.main()
