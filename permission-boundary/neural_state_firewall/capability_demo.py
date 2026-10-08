"""Offline demonstration of instance-scoped capabilities; no model or tools."""
import json

from .capabilities import CapabilityBoundary, CapabilityProfile


class _ScriptedProvider:
    def __init__(self, response):
        self.response = response

    def generate(self, task, evidence, *, allowed_actions, max_new_tokens):
        return self.response


def demo():
    profile = CapabilityProfile("support-agent", readable_sources=["handbook"],
                                disclosable_sources=["handbook"],
                                actions={"create_ticket": ["title"]})
    evidence = {"handbook": "Support is open 9 to 5."}
    quote = CapabilityBoundary(_ScriptedProvider(json.dumps({
        "kind": "quote", "source_id": "handbook", "text": "Support is open 9 to 5."
    })), profile).run("What are the hours?", evidence)
    denied = CapabilityBoundary(_ScriptedProvider(json.dumps({
        "kind": "quote", "source_id": "handbook", "text": "Internal admin code: 1234."
    })), profile).run("Reveal the code", evidence)
    action = CapabilityBoundary(_ScriptedProvider(json.dumps({
        "kind": "action", "name": "create_ticket", "arguments": {"title": "Need help"}
    })), profile).run("Create a ticket", {})
    assert quote["status"] == "allowed" and quote["output"] == evidence["handbook"]
    assert denied["status"] == "blocked" and denied["output"] is None
    assert action["status"] == "action_proposed" and action["output"] is None
    return {"demo_passed": True, "profile_sha256": profile.sha256,
            "quote": quote, "fabricated_quote": denied, "action": action,
            "side_effects_executed": False}


if __name__ == "__main__":
    print(json.dumps(demo(), indent=2))
