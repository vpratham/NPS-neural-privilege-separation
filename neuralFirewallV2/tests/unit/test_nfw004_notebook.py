import ast, json, tempfile, unittest
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[2] / 'experiments/NFW-04_veto_capability_firewall/NFW_004_Veto_Gated_Neural_Privilege_Separation.ipynb'
NB = json.loads(NB_PATH.read_text())
CELLS = {c['id']: ''.join(c['source']) for c in NB['cells']}

class NFW004Test(unittest.TestCase):
    def setUp(self):
        self.ns = {'__name__':'nfw004_test', 'CAPABILITY_TTL_SECONDS':300, 'math':__import__('math')}
        self.ns.update({'base64':__import__('base64'),'csv':__import__('csv'),'gc':__import__('gc'),'hashlib':__import__('hashlib'),'hmac':__import__('hmac'),'io':__import__('io'),'json':__import__('json'),'os':__import__('os'),'re':__import__('re'),'secrets':__import__('secrets'),'sys':__import__('sys'),'tempfile':__import__('tempfile'),'time':__import__('time'),'uuid':__import__('uuid')})
        self.ns.update({'dataclass':__import__('dataclasses').dataclass,'Path':Path})
        self.ns['RUN_DIR'] = Path(tempfile.mkdtemp())
        exec(compile(CELLS['storage'], 'storage', 'exec'), self.ns)
        exec(compile(CELLS['broker'], 'broker', 'exec'), self.ns)
        exec(compile(CELLS['attestation'], 'attestation', 'exec'), self.ns)

    def test_cells_compile_without_saved_outputs(self):
        for c in NB['cells']:
            if c['cell_type'] == 'code':
                ast.parse(''.join(c['source']))
                self.assertEqual(c['outputs'], [])
                self.assertIsNone(c['execution_count'])

    def test_model_claim_is_ignored_and_authority_requires_trusted_token(self):
        B = self.ns['CapabilityBroker']; broker = B(b'x'*32, now=lambda: 100)
        subject = 's'; request = self.ns['model_request']({'tool':'lookup_public_fact','arguments':{'query':'x'},'capability':'admin','signature':'fake'})
        d = broker.authorize(subject=subject, request=request, token=None)
        self.assertFalse(d.allowed); self.assertEqual(d.reason, 'missing_or_untrusted_token')

    def test_valid_scope_veto_expiry_and_replay(self):
        now=[100]; B=self.ns['CapabilityBroker']; broker=B(b'x'*32,now=lambda:now[0]); subject='s'
        token=broker.mint_capability(subject=subject,capability='lookup_public_fact',resources=('query',),ttl_seconds=2)
        req={'tool':'lookup_public_fact','arguments':{'query':'x'}}
        self.assertTrue(broker.authorize(subject=subject,request=req,token=token).allowed)
        self.assertEqual(broker.authorize(subject=subject,request=req,token=token).reason,'replay')
        token=broker.mint_capability(subject=subject,capability='lookup_public_fact',resources=('query',),ttl_seconds=2)
        self.assertEqual(broker.authorize(subject=subject,request=req,token=token,attestation='deny').reason,'neural_veto')
        token=broker.mint_capability(subject=subject,capability='lookup_public_fact',resources=('query',),ttl_seconds=2); now[0]=103
        self.assertEqual(broker.authorize(subject=subject,request=req,token=token).reason,'expired')

    def test_neural_signal_is_veto_only(self):
        compose=self.ns['compose_attestation']
        self.assertEqual(compose('deny'),'deny'); self.assertEqual(compose('approval_required'),'approval_required'); self.assertEqual(compose('allow'),'allow')
        self.assertFalse(self.ns['CapabilityBroker'](b'x'*32).authorize(subject='s',request={'tool':'shell','arguments':{'command':'x'}},token=None).allowed)

if __name__ == '__main__': unittest.main()
