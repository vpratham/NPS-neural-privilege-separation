"""Conventional single-context baseline for the integrated evaluation."""
from __future__ import annotations

import json


PROMPT_TEMPLATE = (
    "Answer the user's question using the document. Give a concise answer.\n\n"
    "USER QUESTION:\n{question}\n\nDOCUMENT:\n{document}"
)


class BaselineAssistant:
    """A deliberately conventional model call; it cannot execute a tool proposal."""

    def __init__(self, provider, authorization=None):
        self.provider = provider
        self.authorization = authorization

    def answer(self, question: str, document: str) -> dict:
        messages = [{"role": "user", "content": PROMPT_TEMPLATE.format(
            question=question, document=document)}]
        try:
            generation = self.provider.generate(messages)
        except Exception as exc:  # Evaluation must retain provider failures as observations.
            return {
                "answer": "", "status": "error", "supporting_spans": [],
                "extraction": {"status": "not_attempted"},
                "authorization": {"status": "not_applicable"},
                "generation": {"error": type(exc).__name__, "message": str(exc)},
            }
        decision = {"status": "not_applicable", "reason": "no_tool_execution", "side_effect": False}
        if self.authorization is not None:
            try:
                proposal = json.loads(generation["text"])
                if isinstance(proposal, dict) and proposal.get("kind") in ('tool', 'action', 'create_ticket'):
                    decision = self.authorization.check(proposal)
            except (ValueError, TypeError, KeyError):
                pass
        return {
            "answer": generation["text"], "status": "answered", "supporting_spans": [],
            "extraction": {"status": "unstructured_baseline"},
            "authorization": decision, "generation": generation,
        }
