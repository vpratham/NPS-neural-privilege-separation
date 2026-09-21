import ast
import json
import tempfile
import unittest
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[2] / 'experiments/NFW-06_adaptive_sandbox_evaluation/NFW_006_Adaptive_Sandboxed_Capability_Firewall.ipynb'
NB = json.loads(NB_PATH.read_text(encoding='utf-8'))
CELLS = {
    cell.get('id', f"cell-{index}"): ''.join(cell['source'])
    for index, cell in enumerate(NB['cells'])
}


class NFW006Test(unittest.TestCase):
    def setUp(self):
        import hashlib, hmac, json as json_module, math, os, random, secrets, sys, tempfile as tempfile_module, time, uuid
        from dataclasses import dataclass
        from pathlib import Path as PathType
        namespace = locals()
        namespace.update({
            'RUN_DIR': Path(tempfile.mkdtemp()),
            'SEED': 7,
            'N_ATTACKS': 20,
            'CHECKPOINT_BATCH': 4,
            'json': json_module,
            'Path': PathType,
            'dataclass': dataclass,
        })
        self.ns = namespace
        exec(compile(CELLS['storage'], 'storage', 'exec'), self.ns)
        exec(compile(CELLS['parser'], 'parser', 'exec'), self.ns)
        exec(compile(CELLS['broker'], 'broker', 'exec'), self.ns)

    def test_cells_compile_without_saved_outputs(self):
        for cell in NB['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])

    def test_surrogate_safe_canonical_digest(self):
        self.assertTrue(self.ns['digest']({'wire': '\ud800'}))

    def test_wire_parser_rejects_authority_and_duplicate_keys(self):
        parse_wire = self.ns['parse_wire']
        with self.assertRaisesRegex(ValueError, 'wire_schema'):
            parse_wire('{"tool":"read_public_fact","arguments":{"key":"public"},"capability":"admin"}')
        with self.assertRaisesRegex(ValueError, 'duplicate_json_key'):
            parse_wire('{"tool":"read_public_fact","tool":"shell","arguments":{"key":"public"}}')

    def test_scope_replay_and_side_effect_boundary(self):
        Sandbox = self.ns['MockSandbox']
        Broker = self.ns['CapabilityBroker']
        sandbox = Sandbox()
        broker = Broker(sandbox, now=lambda: 100)
        subject = 'test-subject'
        request = self.ns['json'].dumps({'tool': 'write_synthetic_record', 'arguments': {'key': 'protected', 'value': 'x'}})
        token = broker.mint(subject, 'write_synthetic_record', 'protected', ('write',))
        self.assertTrue(broker.authorize_and_execute(subject, request, token).allowed)
        self.assertEqual(len(sandbox.records), 1)
        self.assertEqual(broker.authorize_and_execute(subject, request, token).reason, 'replay')
        read_token = broker.mint(subject, 'read_public_fact', 'public', ('read',))
        denied = broker.authorize_and_execute(subject, request, read_token)
        self.assertEqual(denied.reason, 'scope_mismatch')
        self.assertEqual(len(sandbox.records), 1)
        self.assertTrue(broker.audit.verify())

    def test_model_claim_cannot_authorize_or_mutate(self):
        sandbox = self.ns['MockSandbox']()
        broker = self.ns['CapabilityBroker'](sandbox, now=lambda: 100)
        request = self.ns['json'].dumps({'tool': 'write_synthetic_record', 'arguments': {'key': 'protected', 'value': 'x'}, 'capability': 'write'})
        decision = broker.authorize_and_execute('model', request, token=None)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, 'wire_schema')
        self.assertEqual(sandbox.records, [])


if __name__ == '__main__':
    unittest.main()
