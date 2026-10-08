import json
import unittest

from neural_state_firewall.capabilities import CapabilityBoundary, CapabilityProfile, REFUSAL_TEXT


class FakeProvider:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate(self, task, evidence, *, allowed_actions, max_new_tokens):
        self.calls.append((task, evidence, allowed_actions, max_new_tokens))
        return self.response


class CapabilityBoundaryTests(unittest.TestCase):
    def test_read_disclose_and_action_grants_are_separate(self):
        profile = CapabilityProfile("support-agent", readable_sources=["handbook"],
                                    disclosable_sources=[], actions={"send_notice": ["recipient"]})
        provider = FakeProvider('{"kind":"quote","source_id":"handbook","text":"Open at 9."}')
        result = CapabilityBoundary(provider, profile).run("hours", {"handbook": "Open at 9."})
        self.assertEqual((result["status"], result["reason"], result["output"]),
                         ("blocked", "disclosure_not_allowed", None))
        self.assertEqual(provider.calls[0][2], {"send_notice": frozenset({"recipient"})})

    def test_instance_cannot_read_sources_outside_its_profile(self):
        provider = FakeProvider('{"kind":"refusal"}')
        gate = CapabilityBoundary(provider, CapabilityProfile("reader", readable_sources=["public"]))
        result = gate.run("question", {"private": "secret"})
        self.assertEqual(result["reason"], "request_exceeds_instance_capabilities")
        self.assertEqual(provider.calls, [])

    def test_oversized_task_or_evidence_is_rejected_before_provider_call(self):
        provider = FakeProvider('{"kind":"refusal"}')
        gate = CapabilityBoundary(provider, CapabilityProfile("reader", readable_sources=["public"]))
        for task, evidence in (("x" * 16001, {}), ("question", {"public": "x" * 80001})):
            with self.subTest(task_length=len(task), evidence_length=sum(map(len, evidence.values()))):
                self.assertEqual(gate.run(task, evidence)["status"], "blocked")
        self.assertEqual(provider.calls, [])

    def test_only_exact_quotes_from_disclosable_sources_are_released(self):
        profile = CapabilityProfile("reader", readable_sources=["handbook"],
                                    disclosable_sources=["handbook"])
        raw = json.dumps({"kind": "quote", "source_id": "handbook", "text": "Open at 9."})
        result = CapabilityBoundary(FakeProvider(raw), profile).run("hours", {"handbook": "Open at 9. Close at 5."})
        self.assertEqual((result["status"], result["output"], result["instance_id"]),
                         ("allowed", "Open at 9.", "reader"))
        fabricated = CapabilityBoundary(FakeProvider(raw.replace("Open at 9.", "Open at 10.")), profile)
        self.assertEqual(fabricated.run("hours", {"handbook": "Open at 9."})["reason"],
                         "quote_not_in_authorized_evidence")

    def test_action_must_be_granted_with_exact_argument_names_and_is_never_executed(self):
        profile = CapabilityProfile("writer", actions={"write_note": ["note_id", "content"]})
        raw = '{"kind":"action","name":"write_note","arguments":{"note_id":"n1","content":"hello"}}'
        result = CapabilityBoundary(FakeProvider(raw), profile).run("save", {})
        self.assertEqual((result["status"], result["reason"]),
                         ("action_proposed", "broker_authorization_required"))
        self.assertIsNone(result["output"])
        bad = raw.replace(',"content":"hello"', ',"content":"hello","admin":true')
        denied = CapabilityBoundary(FakeProvider(bad), profile).run("save", {})
        self.assertEqual(denied["reason"], "action_arguments_not_allowed")

    def test_unstructured_prose_duplicate_keys_and_extra_fields_fail_closed(self):
        profile = CapabilityProfile("reader", readable_sources=["handbook"],
                                    disclosable_sources=["handbook"])
        for raw in ("answer is 9", '{"kind":"refusal","kind":"quote"}',
                    '{"kind":"refusal","text":"secret"}'):
            result = CapabilityBoundary(FakeProvider(raw), profile).run("question", {"handbook": "9"})
            self.assertIn(result["status"], ("blocked", "error"))
            self.assertIsNone(result["output"])

    def test_host_controlled_refusal_and_invalid_profile(self):
        result = CapabilityBoundary(FakeProvider('{"kind":"refusal"}'),
                                    CapabilityProfile("reader")).run("question", {})
        self.assertEqual((result["status"], result["output"]), ("refused", REFUSAL_TEXT))
        with self.assertRaisesRegex(ValueError, "disclosure_requires_read_permission"):
            CapabilityProfile("reader", disclosable_sources=["private"])
        profile = CapabilityProfile("reader", readable_sources=["public"])
        with self.assertRaises(AttributeError):
            profile.instance_id = "admin"


if __name__ == "__main__":
    unittest.main()
