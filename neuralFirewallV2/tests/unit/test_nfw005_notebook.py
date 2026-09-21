import ast, json, tempfile, unittest
from pathlib import Path
NB_PATH=Path(__file__).resolve().parents[2]/'experiments/NFW-05_adversarial_capability_firewall/NFW_005_Reproducible_Adversarial_Capability_Firewall.ipynb'
NB=json.loads(NB_PATH.read_text()); CELLS={c['id']:''.join(c['source']) for c in NB['cells']}
class NFW005Test(unittest.TestCase):
 def setUp(self):
  import base64,hashlib,hmac,json,math,os,random,re,secrets,sys,tempfile,time,uuid
  self.ns={'__name__':'nfw005_test','CHECKPOINT_BATCH':4,'SEED':7,'N_RANDOM_CASES':40,'CAPABILITY_TTL_SECONDS':300,'RUN_DIR':Path(tempfile.mkdtemp()),'MAX_WIRE_BYTES':8192,'MAX_STRING':1024}
  self.ns.update(locals()); self.ns['dataclass']=__import__('dataclasses').dataclass; self.ns['Path']=Path
  self.ns.update({'canonical':None})
  exec(compile(CELLS['storage'],'storage','exec'),self.ns); exec(compile(CELLS['broker'],'broker','exec'),self.ns)
 def test_cells_compile_clean(self):
  for c in NB['cells']:
   if c['cell_type']=='code': ast.parse(''.join(c['source'])); self.assertEqual(c['outputs'],[]); self.assertIsNone(c['execution_count'])
 def test_wire_parser_rejects_duplicate_nonfinite_and_authority_claims(self):
  self.assertTrue(self.ns['digest']({'wire': '\ud800'}))
  p=self.ns['parse_wire']
  with self.assertRaisesRegex(ValueError,'duplicate_json_key'): p('{"tool":"a","tool":"b","arguments":{}}')
  with self.assertRaisesRegex(ValueError,'nonfinite_json'): p('{"tool":"lookup_public_fact","arguments":{"query":NaN}}')
  with self.assertRaisesRegex(ValueError,'wire_schema'): p('{"tool":"lookup_public_fact","arguments":{},"capability":"admin"}')
 def test_broker_scope_replay_rotation_audit(self):
  B=self.ns['CapabilityBroker']; b=B(keys={1:b'x'*32},now=lambda:100); s='u'; req='{"tool":"lookup_public_fact","arguments":{"query":"x"}}'
  t=b.mint_capability(s,'lookup_public_fact',('query',),ttl=5); self.assertTrue(b.authorize(s,req,t).allowed); self.assertEqual(b.authorize(s,req,t).reason,'replay')
  t=b.mint_capability(s,'lookup_public_fact',('query',)); b.rotate_key(2,b'y'*32); self.assertTrue(b.authorize(s,req,t).allowed)
  t=b.mint_capability(s,'lookup_public_fact',('query',)); self.assertTrue(b.authorize(s,req,t).allowed); self.assertTrue(b.audit.verify())
 def test_model_cannot_authorize_without_trusted_token(self):
  b=self.ns['CapabilityBroker'](keys={1:b'x'*32},now=lambda:100)
  req='{"tool":"lookup_public_fact","arguments":{"query":"x"},"capability":"admin"}'
  self.assertFalse(b.authorize('u',req,None).allowed)
 def test_fuzz_progress_keeps_prior_outcomes(self):
  # Regression for disconnect checkpoint semantics in the actual storage helper.
  save=self.ns['save_stage']; load=self.ns['load_stage']; binding='fuzz-binding'
  prior={'next':4,'outcomes':[{'id':i} for i in range(4)]}
  save('fuzz_progress.json',prior,binding,replace=True)
  loaded=load('fuzz_progress.json',binding)
  later={'next':8,'outcomes':list(loaded['outcomes'])+[{'id':i} for i in range(4,8)]}
  save('fuzz_progress.json',later,binding,replace=True)
  self.assertEqual(load('fuzz_progress.json',binding),later)
if __name__=='__main__': unittest.main()
