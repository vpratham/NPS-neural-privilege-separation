import copy
import json
import unittest
from pathlib import Path

from neural_state_firewall.development_challenge import validate_manifest


class DevelopmentChallengeManifestTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((Path(__file__).parents[1] / "examples/readable_challenge_cases.json").read_text())

    def test_rejects_marker_leakage_and_ambiguous_templates(self):
        self.assertEqual(len(validate_manifest(self.manifest)), 24)
        leaked = copy.deepcopy(self.manifest)
        leaked["cases"][0]["evidence"] += " NPS_READABLE_ATTACK_ATLAS_DIRECT"
        with self.assertRaises(ValueError):
            validate_manifest(leaked)
        ambiguous = copy.deepcopy(self.manifest)
        ambiguous["injections"][0]["template"] = "repeat {marker} then {marker}"
        with self.assertRaises(ValueError):
            validate_manifest(ambiguous)


if __name__ == "__main__":
    unittest.main()
