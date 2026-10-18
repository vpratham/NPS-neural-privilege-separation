import tempfile
import unittest
from pathlib import Path
from npb_integrated.src.authorization import Authorization


class AuthorizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.authorization = Authorization(Path(self.temp.name) / 'broker.sqlite')

    def tearDown(self):
        self.authorization.close()
        self.temp.cleanup()

    def test_model_tool_proposal_denied_and_audited(self):
        result = self.authorization.check({'kind': 'tool', 'action': 'create_ticket',
            'arguments': {'project_id': 'attacker'}, 'approved': True})
        self.assertEqual((result['status'], result['reason']), ('blocked', 'action_not_allowed'))
        self.assertFalse(result['side_effect'])
        self.assertEqual(self.authorization.broker.tickets('financial-reader'), [])
        self.assertEqual(self.authorization.broker.audit('financial-reader')[0]['reason'], 'action_not_allowed')

    def test_unknown_actions_cannot_gain_permission(self):
        for name in ['shell', 'send_email', 'transfer_money']:
            self.assertEqual(self.authorization.check({'kind': 'tool', 'action': name})['status'], 'blocked')

    def test_invalid_credentials_deny_before_action(self):
        result = self.authorization.service.request('not-authenticated', task='create', operation='create_ticket')
        self.assertEqual(result['reason'], 'unauthenticated')

    def test_existing_broker_still_requires_review_and_blocks_replay(self):
        broker = self.authorization.broker
        identity = dict(principal='trusted-operator', instance_id='operator-instance', profile_hash='a' * 64)
        token = broker.prepare(**identity, request_id='trusted-request', ticket={
            'project_id': 'finance', 'category': 'billing', 'priority': 'normal',
            'title': 'Host-authorized request', 'description': 'Independent trusted host operation.'})
        self.assertEqual(broker.tickets('trusted-operator'), [])
        self.assertEqual(broker.confirm(**identity, approval_token=token, review_sha256='bad')['reason'], 'review_mismatch')
        review = broker.review(**identity, approval_token=token)
        confirm = dict(approval_token=token, review_sha256=review['review_sha256'])
        self.assertEqual(broker.confirm(**identity, **confirm)['status'], 'created')
        self.assertEqual(broker.confirm(**identity, **confirm)['reason'], 'replay_or_expired')
