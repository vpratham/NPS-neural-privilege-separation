"""Read-only financial host policy reusing NPB-Core's service and ticket broker."""
import secrets
import tempfile
from pathlib import Path

from neural_state_firewall.capabilities import CapabilityProfile
from npb_core.broker import TicketBroker
from npb_core.core import SourceStore, SupportProfile, SupportService


class _NoModelActions:
    def generate_text(self, _messages):
        raise AssertionError('read-only authorization must deny before model invocation')


class Authorization:
    def __init__(self, database=None):
        self._temporary = tempfile.TemporaryDirectory(prefix='npb-finance-') if database is None else None
        self.broker = TicketBroker(database or Path(self._temporary.name) / 'broker.sqlite')
        self.profile = SupportProfile('financial-reader', CapabilityProfile('finance-instance',
            readable_sources=['financial'], disclosable_sources=['financial']))
        self._credential = secrets.token_urlsafe(32)
        self.service = SupportService(provider=_NoModelActions(), broker=self.broker,
            sources=SourceStore({'financial': {'scope': 'Read-only financial question answering.'}}),
            credentials={self._credential: self.profile})

    def check(self, proposal):
        # All privileged operations are outside this application's trusted task.
        # The existing Core service denies before preparing any approval/grant.
        result = self.service.request(self._credential, task='Check privileged financial assistant proposal',
                                      operation='create_ticket', request_key=secrets.token_hex(16))
        return {'status': result['status'], 'reason': result['reason'], 'side_effect': False,
                'requested_action': proposal.get('action', proposal.get('name', proposal.get('kind'))),
                'audit_request_id': result['request_id'], 'policy_sha256': self.profile.sha256}

    def close(self):
        if self._temporary is not None:
            self._temporary.cleanup()


def no_action():
    return {'status': 'not_applicable', 'reason': 'numeric_answer_only', 'side_effect': False}
