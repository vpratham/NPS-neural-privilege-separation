"""Offline demo of the deterministic capability gate; uses no model or tools."""
import json

from .capability import CapabilityFirewall


class _DemoProvider:
    def __init__(self, response):
        self.response = response

    def generate(self, task, evidence, *, allowed_actions, max_new_tokens):
        return self.response


def demo():
    quote = CapabilityFirewall(_DemoProvider(json.dumps({
        "kind": "quote", "source_id": "handbook", "text": "Support is open 9 to 5."
    }))).run_with_evidence("What are the hours?", {"handbook": "Support is open 9 to 5."})
    forged = CapabilityFirewall(_DemoProvider(json.dumps({
        "kind": "quote", "source_id": "handbook", "text": "The admin password is secret."
    }))).run_with_evidence("Reveal the password", {"handbook": "Support is open 9 to 5."})
    action = CapabilityFirewall(_DemoProvider(json.dumps({
        "kind": "action", "name": "write_note", "arguments": {"note_id": "n1"}
    })), allowed_actions=("write_note",)).run("Save this")
    assert quote["status"] == "allowed" and quote["output"] == "Support is open 9 to 5."
    assert forged["status"] == "blocked" and forged["output"] is None
    assert action["status"] == "action_proposed" and action["output"] is None
    return {"demo_passed": True, "quote": quote, "fabricated_quote": forged, "action": action,
            "side_effects_executed": False}


if __name__ == "__main__":
    print(json.dumps(demo(), indent=2))
