"""The cloud preflight cannot turn incomplete mechanics into security approval."""
import copy
import unittest
from unittest.mock import patch

from docs.security_hardening.colab_day1 import main, summarize_gpu


class ColabPreflightTests(unittest.TestCase):
    def test_completion_errors_and_structural_failures_block_mechanics_gate(self):
        attempt = {"ended_with_eos": True, "error_type": None}
        report = {"structural_checks_passed": True, "rows": [{
            "arms": {a: copy.deepcopy(attempt) for a in ("ordinary", "isolated_memory", "read_permissions")},
            "denied_variant": copy.deepcopy(attempt), "structural_check_passed": True}]}
        self.assertTrue(summarize_gpu(report)["development_mechanics_passed"])
        self.assertFalse(summarize_gpu(report)["semantic_security_approved"])
        for mutate in (lambda r: r["rows"][0]["denied_variant"].update(ended_with_eos=False),
                       lambda r: r["rows"][0]["arms"]["ordinary"].update(error_type="TimeoutError"),
                       lambda r: r["rows"][0].update(structural_check_passed=False),
                       lambda r: r.update(rows=[])):
            broken = copy.deepcopy(report)
            mutate(broken)
            self.assertFalse(summarize_gpu(broken)["development_mechanics_passed"])

    def test_mac_cannot_start_remote_model_job(self):
        with patch("docs.security_hardening.colab_day1.platform.system", return_value="Darwin"):
            with self.assertRaisesRegex(RuntimeError, "Colab Linux"):
                main()
