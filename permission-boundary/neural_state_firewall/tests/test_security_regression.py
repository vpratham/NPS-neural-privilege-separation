"""A completed attack response must not pass the known-failure regression gate."""
import unittest

from neural_state_firewall.security_regression import CASES, regression_gate


class SecurityRegressionTests(unittest.TestCase):
    def report(self):
        return {"rows": [{"case_id": case_id, "condition": condition, "status": "allowed",
                          "all_exact_attack_indicators_present": False}
                         for case_id in CASES for condition in ("clean", "attack_readable")]}

    def test_completed_attack_is_failure_and_block_is_not_a_semantic_certificate(self):
        report = self.report()
        self.assertTrue(regression_gate(report)["passed"])
        self.assertFalse(regression_gate(report)["semantic_security_approved"])
        report["rows"][1]["all_exact_attack_indicators_present"] = True
        result = regression_gate(report)
        self.assertFalse(result["passed"])
        self.assertEqual(result["known_payload_released"], [report["rows"][1]["case_id"]])
        report["rows"][1]["status"] = "blocked"
        self.assertTrue(regression_gate(report)["passed"])
        report["rows"][0]["status"] = "blocked"
        self.assertFalse(regression_gate(report)["passed"])

    def test_incomplete_and_error_responses_are_not_security_successes(self):
        for status in ("incomplete", "error"):
            for index in (0, 1):
                report = self.report()
                report["rows"][index]["status"] = status
                verdict = regression_gate(report)
                self.assertFalse(verdict["passed"])
                self.assertEqual(len(verdict["unavailable_responses"]), 1)
                self.assertEqual(verdict["known_payload_released"], [])

    def test_missing_duplicate_or_malformed_accounting_fails_closed(self):
        for report in (None, {}, {"rows": "bad"}, {"rows": 1}, {"rows": [None]}, self.report()):
            if isinstance(report, dict) and isinstance(report.get("rows"), list) and len(report["rows"]) > 1:
                report["rows"][0] = report["rows"][1].copy()
            self.assertFalse(regression_gate(report)["passed"])
        report = self.report()
        report["rows"][1]["all_exact_attack_indicators_present"] = "false"
        self.assertFalse(regression_gate(report)["passed"])


if __name__ == "__main__":
    unittest.main()
