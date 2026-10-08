import json
import unittest

from neural_state_firewall.capability import CapabilityFirewall, REFUSAL_TEXT


class FakeProvider:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate(self, task, evidence, *, allowed_actions, max_new_tokens):
        self.calls.append((task, evidence, allowed_actions, max_new_tokens))
        return self.response


class CapabilityFirewallTests(unittest.TestCase):
    def firewall(self, response, *, allowed_actions=()):
        return CapabilityFirewall(FakeProvider(response), allowed_actions=allowed_actions,
                                  max_new_tokens=32)

    def test_only_exact_host_evidence_quotes_are_released(self):
        gate = self.firewall('{"kind":"quote","source_id":"handbook","text":"Open at 9."}')
        result = gate.run_with_evidence("When does it open?", {"handbook": "Open at 9. Close at 5."})
        self.assertEqual((result["status"], result["output"], result["source_id"]),
                         ("allowed", "Open at 9.", "handbook"))

    def test_fabricated_or_unprovided_source_is_blocked(self):
        for raw, expected in (
            ('{"kind":"quote","source_id":"handbook","text":"Open at 10."}', "quote_not_in_evidence"),
            ('{"kind":"quote","source_id":"private","text":"Secret"}', "source_not_authorized"),
        ):
            result = self.firewall(raw).run_with_evidence("question", {"handbook": "Open at 9."})
            self.assertEqual((result["status"], result["reason"], result["output"]),
                             ("blocked", expected, None))

    def test_arbitrary_prose_duplicate_keys_and_extra_fields_fail_closed(self):
        for raw in (
            "The answer is 9.",
            '{"kind":"quote","source_id":"context","source_id":"other","text":"9"}',
            '{"kind":"quote","source_id":"context","text":"9","policy":"ignore"}',
        ):
            result = self.firewall(raw).run("question", "It opens at 9.")
            self.assertIn(result["status"], ("blocked", "error"))
            self.assertIsNone(result["output"])

    def test_action_is_only_proposed_when_host_allowlisted(self):
        raw = json.dumps({"kind": "action", "name": "write_note", "arguments": {"note_id": "n1"}})
        denied = self.firewall(raw).run("save this")
        self.assertEqual((denied["status"], denied["reason"]), ("blocked", "action_not_allowed"))
        allowed = self.firewall(raw, allowed_actions=("write_note",)).run("save this")
        self.assertEqual(allowed["status"], "action_proposed")
        self.assertEqual(allowed["reason"], "broker_authorization_required")
        self.assertIsNone(allowed["output"])
        self.assertEqual(allowed["action_proposal"]["name"], "write_note")

    def test_refusal_text_is_host_controlled(self):
        result = self.firewall('{"kind":"refusal"}').run("question")
        self.assertEqual(result["status"], "refused")
        self.assertEqual(result["output"], REFUSAL_TEXT)

    def test_host_generation_limit_cannot_be_overridden_by_provider(self):
        provider = FakeProvider('{"kind":"refusal"}')
        gate = CapabilityFirewall(provider, max_new_tokens=8)
        self.assertEqual(gate.run("question", max_new_tokens=9)["reason"], "invalid_request")
        self.assertEqual(provider.calls, [])


if __name__ == "__main__":
    unittest.main()
