"""Offline behavioral regression tests for the NFW-003 Colab notebook.

These execute the notebook's real helper cells without Google Drive, Hugging Face,
or a GPU.  Run with:
  python3 -m unittest neuralFirewallV2.tests.unit.test_nfw003_notebook -v
"""
import ast
import csv
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import unittest
import warnings
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

NOTEBOOK = Path(__file__).resolve().parents[2] / 'experiments/NFW-03_behavioral_proxy/NFW_003_Behavioral_Proxy_Firewall.ipynb'
NB = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
CELLS = {cell['id']: ''.join(cell['source']) for cell in NB['cells']}


def namespace(directory):
    ns = dict(globals())
    ns.update(RUN_DIR=Path(directory), manifest={'stages': {}}, __name__='nfw003_test_helpers')
    for cell in ['storage', 'data-helpers', 'statistics', 'audit-helpers']:
        exec(compile(CELLS[cell], cell, 'exec'), ns)
    return ns


def raw_row(source, index, response_harmful, *, prompt=None, response=None,
            prompt_harmful=0, prompt_adversarial=0, language='en'):
    return {
        'prompt': prompt if prompt is not None else f'prompt {source} {index}',
        'response': response if response is not None else f'response {source} {index}',
        'response_harmful': response_harmful,
        'source': source,
        'language': language,
        'prompt_harmful': prompt_harmful,
        'prompt_adversarial': prompt_adversarial,
        'category': 'test',
    }


class NFW003NotebookTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ns = namespace(self.temp.name)
        self.cfg = {
            'language': 'en', 'max_prompt_chars': 1000, 'max_response_chars': 1000,
            'max_sources': 64, 'per_source_response_class': 3,
        }

    def rows_with_source_class_coverage(self, sources=12):
        rows = []
        for source in range(sources):
            for label in (0, 1):
                candidate = self.ns['candidate'](raw_row(f'source-{source}', label, label), self.cfg, Counter())
                rows.append(candidate)
        return rows

    def test_notebook_cells_compile_and_are_clean(self):
        self.assertEqual(len(CELLS), len(NB['cells']))
        for cell in NB['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), cell['id'], 'exec')
                self.assertEqual(cell['outputs'], [], cell['id'])
                self.assertIsNone(cell['execution_count'], cell['id'])

    def test_stage_envelope_checksum_binding_and_crash_orphan_adoption(self):
        ns = self.ns
        ns['save_stage']('stage.json', {'x': 1}, 'binding-a')
        self.assertEqual(ns['load_stage']('stage.json', 'binding-a'), {'x': 1})
        with self.assertRaisesRegex(RuntimeError, 'binding mismatch'):
            ns['load_stage']('stage.json', 'binding-b')
        ns['mark_stage']('stage', 'stage.json')  # result survived before manifest commit
        ns['mark_stage']('stage', 'stage.json')  # repeated reconnect is idempotent
        path = Path(self.temp.name) / 'stage.json'
        envelope = json.loads(path.read_text())
        envelope['payload']['x'] = 2
        path.write_text(json.dumps(envelope), encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError, 'checksum mismatch'):
            ns['load_stage']('stage.json', 'binding-a')

    def test_checkpoint_skips_completed_work_and_refuses_changed_contract(self):
        calls = []
        def produce():
            calls.append('ran')
            return {'complete': True}
        self.assertEqual(self.ns['checkpoint']('features/prompt/a.json', 'v1', produce), {'complete': True})
        self.assertEqual(self.ns['checkpoint']('features/prompt/a.json', 'v1', lambda: self.fail('reran checkpoint')), {'complete': True})
        self.assertEqual(calls, ['ran'])
        with self.assertRaisesRegex(RuntimeError, 'binding mismatch'):
            self.ns['checkpoint']('features/prompt/a.json', 'v2', produce)

    def test_candidate_rejects_missing_labels_and_sampler_is_bounded_order_invariant(self):
        counts = Counter()
        self.assertIsNone(self.ns['candidate'](raw_row('x', 0, None), self.cfg, counts))
        self.assertEqual(counts['missing_response_label'], 1)
        raw = [raw_row('s', i, i % 2) for i in range(50)]
        def sample(values):
            pools = {}
            for value in values:
                row = self.ns['candidate'](value, self.cfg, Counter())
                self.ns['update_pool'](pools, row, self.cfg)
            return self.ns['finish_sample'](pools)
        a, excluded_a = sample(raw)
        b, excluded_b = sample(reversed(raw))
        self.assertEqual(a, b)
        self.assertEqual(excluded_a, excluded_b)
        self.assertLessEqual(len(a), 2 * self.cfg['per_source_response_class'])

    def test_sampling_quarantines_cross_source_and_conflicting_prompt_duplicates(self):
        cfg = dict(self.cfg, per_source_response_class=10)
        pools = {}
        for raw in [
            raw_row('one', 0, 0, prompt='duplicate'),
            raw_row('two', 0, 0, prompt=' DUPLICATE '),
            raw_row('three', 0, 0, prompt='conflict'),
            raw_row('three', 1, 1, prompt='conflict'),
            raw_row('four', 0, 1, prompt='unique'),
        ]:
            self.ns['update_pool'](pools, self.ns['candidate'](raw, cfg, Counter()), cfg)
        rows, excluded = self.ns['finish_sample'](pools)
        self.assertEqual([row['prompt'] for row in rows], ['unique'])
        self.assertEqual(len(excluded), 2)
        self.assertTrue(all(e['reason'] == 'sampled_cross_source_or_response_label_conflict' for e in excluded))

    def test_source_budget_is_deterministic_and_bounded(self):
        cfg = dict(self.cfg, max_sources=5, per_source_response_class=2)
        raw = [raw_row(f'source-{source}', i, i % 2) for source in range(50) for i in range(5)]
        def sampled(values):
            pools = {}
            for value in values:
                self.ns['update_pool'](pools, self.ns['candidate'](value, cfg, Counter()), cfg)
            return self.ns['finish_sample'](pools)[0]
        a, b = sampled(raw), sampled(reversed(raw))
        self.assertEqual(a, b)
        self.assertLessEqual(len({row['source'] for row in a}), 5)
        self.assertLessEqual(len(a), 5 * 2 * 2)

    def test_source_disjoint_splits_and_leakage_and_minimum_rejection(self):
        rows = self.rows_with_source_class_coverage(16)
        doc = self.ns['make_splits'](rows, 42, 1)
        self.ns['validate_splits'](doc, rows, 1)
        # A duplicate ID is leakage even when source IDs happen to differ in other rows.
        bad = {name: list(ids) for name, ids in doc.items()}
        bad['final'][0] = bad['train'][0]
        with self.assertRaisesRegex(ValueError, 'partition'):
            self.ns['validate_splits'](bad, rows, 1)
        with self.assertRaisesRegex(RuntimeError, 'Need >=8'):
            self.ns['make_splits'](self.rows_with_source_class_coverage(7), 42, 1)

    def test_threshold_selection_ties_nonfinite_values_and_metric_denominators(self):
        ns = self.ns
        selected = ns['select_threshold']([1., 1., 1., 1.], [0, 0, 1, 1], .80, .10)
        self.assertTrue(math.isfinite(selected['threshold']))
        self.assertEqual(selected['safe_response_block_rate'], 0.)
        self.assertFalse(selected['target_recall_met'])
        with self.assertRaises(ValueError):
            ns['select_threshold']([float('nan'), 1], [0, 1], .8, .1)
        rows = [
            {'response_harmful': 1, 'prompt_harmful': 1, 'prompt_adversarial': 0},
            {'response_harmful': 0, 'prompt_harmful': 0, 'prompt_adversarial': 0},
        ]
        metrics = ns['proxy_metrics'](rows, [2., 0.], 1.)
        self.assertEqual(metrics['released_harmful_n'], 0)
        self.assertEqual(metrics['safe_response_block_rate'], 0.)
        self.assertEqual(metrics['benign_prompt_false_block_rate'], 0.)
        self.assertEqual(metrics['task_success'], None)

    def test_csv_formula_defense_and_request_bound_review_deduplication(self):
        ns = self.ns
        self.assertEqual(ns['spreadsheet_text']('=1+1'), 'TEXT: =1+1')
        rows = [
            {'id': 'a', 'prompt': 'request A', 'prompt_harmful': 0, 'prompt_adversarial': 0},
            {'id': 'b', 'prompt': 'request B', 'prompt_harmful': 0, 'prompt_adversarial': 0},
            {'id': 'c', 'prompt': 'request A', 'prompt_harmful': 0, 'prompt_adversarial': 0},
        ]
        responses = [
            {'id': 'a', 'condition': 'baseline', 'released': True, 'text': 'same'},
            {'id': 'b', 'condition': 'baseline', 'released': True, 'text': 'same'},
            {'id': 'c', 'condition': 'prompt_gate', 'released': True, 'text': 'same'},
            {'id': 'a', 'condition': 'prompt_gate', 'released': False, 'text': ''},
        ]
        items, keys = ns['make_review_items'](rows, responses, 'rubric-v1')
        self.assertEqual(len(items), 2)  # same request/response shares a label; distinct request does not
        self.assertEqual(keys[('a', 'baseline')], keys[('c', 'prompt_gate')])
        self.assertNotEqual(keys[('a', 'baseline')], keys[('b', 'baseline')])
        path = Path(self.temp.name) / 'review.csv'
        ns['write_review'](path, items)
        found, pending = ns['read_review'](path, items)
        self.assertEqual(set(found), set(items)); self.assertEqual(set(pending), set(items))
        # Existing review file is intentionally not overwritten after a disconnect.
        original = path.read_text(encoding='utf-8')
        ns['write_review'](path, {})
        self.assertEqual(path.read_text(encoding='utf-8'), original)

    def test_review_validation_and_paired_audit_metrics(self):
        ns = self.ns
        rows = [
            {'id': 'safe', 'prompt': 'safe', 'prompt_harmful': 0, 'prompt_adversarial': 0},
            {'id': 'harm', 'prompt': 'harm', 'prompt_harmful': 1, 'prompt_adversarial': 0},
        ]
        responses = [
            {'id': ident, 'condition': condition, 'released': released, 'text': text}
            for ident, condition, released, text in [
                ('safe', 'baseline', True, 'answer'), ('safe', 'prompt_gate', False, ''),
                ('harm', 'baseline', True, 'bad'), ('harm', 'prompt_gate', True, 'refusal'),
            ]
        ]
        items, keys = ns['make_review_items'](rows, responses, 'v1')
        path = Path(self.temp.name) / 'review.csv'
        ns['write_review'](path, items)
        # Invalid labels are rejected rather than silently coerced.
        with path.open('w', newline='', encoding='utf-8') as f:
            fields = ['review_id', 'request', 'response', 'harmful_assistance', 'task_success', 'reviewer']
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
            for key, item in items.items():
                writer.writerow(item | {'harmful_assistance': '2', 'task_success': '1', 'reviewer': 'reviewer'})
        with self.assertRaises(ValueError): ns['read_review'](path, items)
        # Restore a complete, request-bound review table.
        with path.open('w', newline='', encoding='utf-8') as f:
            fields = ['review_id', 'request', 'response', 'harmful_assistance', 'task_success', 'reviewer']
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
            for key, item in items.items():
                writer.writerow(item | {'harmful_assistance': '0', 'task_success': '1', 'reviewer': 'reviewer'})
        reviewed, pending = ns['read_review'](path, items)
        self.assertEqual(pending, [])
        values = ns['audit_metrics'](rows, responses, items, keys, reviewed)
        self.assertEqual(values['prompt_gate']['benign_task_success'], 0.)
        self.assertEqual(values['baseline']['benign_task_success'], 1.)
        self.assertEqual(values['prompt_gate']['blocked_n'], 1)
        with self.assertRaisesRegex(ValueError, 'Incomplete audit pairs'):
            ns['audit_metrics'](rows, responses[:-1], items, keys, reviewed)

    def test_capability_gate_is_fail_closed_for_malformed_or_escalating_requests(self):
        gate = self.ns['capability_gate']
        trusted = {'lookup_public_fact'}
        self.assertTrue(gate({'tool': 'lookup_public_fact', 'arguments': {'query': 'neuron'}}, trusted))
        bad = [
            None, {}, {'tool': 'shell', 'arguments': {'command': 'x'}},
            {'tool': 'lookup_public_fact', 'arguments': {'query': 'x', 'grant': 'shell'}},
            {'tool': 'lookup_public_fact', 'arguments': {'query': ''}},
            {'tool': 'lookup_public_fact', 'arguments': {'query': 1}},
            {'tool': 'lookup_public_fact', 'arguments': {'query': 'x' * 201}},
        ]
        self.assertTrue(all(not gate(request, trusted) for request in bad))
        self.assertFalse(gate({'tool': 'lookup_public_fact', 'arguments': {'query': 'x'}}, set()))

    def test_feature_helper_uses_cached_checkpoint_and_rejects_corrupt_layer_payload(self):
        # Execute the real model-helper cell with a tiny tokenizer and fake extraction backend.
        ns = self.ns
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs): return [1, len(messages[0]['content'])]
            def encode(self, text, **kwargs): return [2, len(text)]
        ns.update(tokenizer=Tokenizer(), CONFIG={'layers': [0, 1], 'max_pair_tokens': 10},
                  FEATURE_BINDING='feature-v1', model_hidden_size=2,
                  extract_ids=lambda ids, layers: {str(layer): [float(layer), 1.] for layer in layers})
        module = ast.parse(CELLS['model-helpers'])
        selected = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in {'prompt_ids', 'pair_ids', 'feature'}]
        exec(compile(ast.Module(body=selected, type_ignores=[]), 'model-helper-subset', 'exec'), ns)
        row = {'id': 'r', 'prompt': 'hi', 'response': 'there'}
        first = ns['feature'](row, 'prompt')
        self.assertEqual(first, {'0': [0., 1.], '1': [1., 1.]})
        ns['extract_ids'] = lambda *_: self.fail('cached feature recomputed')
        self.assertEqual(ns['feature'](row, 'prompt'), first)
        path = Path(self.temp.name) / 'features/prompt/r.json'
        env = json.loads(path.read_text()); env['payload']['0'] = [0.] ; env['payload_sha256'] = ns['digest'](env['payload'])
        path.write_text(json.dumps(env), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Feature corruption'):
            ns['feature'](row, 'prompt')


    def test_resumable_stream_scan_executes_actual_notebook_cell(self):
        """A simulated disconnect resumes from the last committed IterableDataset state."""
        ns = self.ns
        records = [raw_row(f'source-{source}', label, label) for source in range(8) for label in (0, 1)]
        class Stream:
            def __init__(self, values, fail_at=None): self.values, self.position, self.fail_at = values, 0, fail_at
            def __iter__(self):
                while self.position < len(self.values):
                    if self.fail_at is not None and self.position == self.fail_at:
                        raise ConnectionError('simulated Colab disconnect')
                    value = self.values[self.position]
                    self.position += 1
                    yield value
            def state_dict(self): return {'position': self.position}
            def load_state_dict(self, state): self.position = state['position']
        streams = [Stream(records, fail_at=3), Stream(records)]
        ns.update(
            MODE='run', BASE_BINDING='base-binding', HF_TOKEN='not-used',
            manifest={'dataset_revision': 'pinned', 'stages': {}},
            CONFIG=dict(self.cfg, dataset_id='offline', dataset_split='train', dataset_revision='pinned',
                        scan_checkpoint_rows=2),
            load_dataset=lambda *args, **kwargs: streams.pop(0),
        )
        with self.assertRaisesRegex(ConnectionError, 'disconnect'):
            exec(compile(CELLS['sample'], 'sample-disconnect', 'exec'), ns)
        progress = ns['load_stage']('scan_progress.json', 'base-binding')
        self.assertEqual(progress['counts']['scanned'], 2)
        self.assertEqual(progress['stream_state'], {'position': 2})
        exec(compile(CELLS['sample'], 'sample-resume', 'exec'), ns)
        sample = ns['load_stage']('dataset_candidates.json', 'base-binding')
        self.assertEqual(sample['counts']['scanned'], len(records))
        self.assertEqual(len(sample['records']), len(records))
        self.assertIn('dataset_candidates.json', ns['manifest']['stages'])

    def test_synthetic_fit_evaluate_audit_and_report_cells_are_resumable(self):
        """Run real stage cells end-to-end with deterministic in-memory features, no GPU/HF."""
        import gc
        ns = self.ns
        # Four source-disjoint-like partitions, each with both historical-response labels.
        splits = {}
        for part, offset in zip(['train', 'development', 'calibration', 'final'], range(4)):
            splits[part] = [
                {'id': f'{part}-{label}', 'source': f'{part}-source-{label}',
                 'prompt': f'{"unsafe" if label else "ordinary"} request {offset}-{label}',
                 'response': f'stored response {label}', 'response_harmful': label,
                 'prompt_harmful': label, 'prompt_adversarial': 0,
                 'metadata': {}}
                for label in (0, 1)
            ]
        # Replicate each label to make the text model's feature matrix non-degenerate.
        for part, rows in splits.items():
            splits[part] = [dict(row, id=row['id'] + f'-{copy}', prompt=row['prompt'] + f' variant {copy}')
                            for row in rows for copy in range(3)]
        config = {
            'layers': [0, 1], 'Cs': [0.1, 1.0], 'seed': 42, 'keywords': ['unsafe'],
            'target_recall': .80, 'max_calibration_fpr': .10, 'fpr_grid': [.05, .10],
            'bootstrap_replicates': 8, 'continuation': False, 'audit_n': 2,
            'max_new_tokens': 4, 'review_rubric': 'synthetic-v1',
        }
        ns.update(gc=gc, MODE='run', RUN_ID='synthetic', CONFIG=config, splits=splits, FEATURE_BINDING='feature-binding',
                  manifest={'run_id': 'synthetic', 'dataset_revision': 'offline-revision', 'stages': {}},
                  RUN_GENERATION_AUDIT=False, RUN_STRESS_TESTS=False, stress=None, comparison=None,
                  row_by_id={row['id']: row for part in splits.values() for row in part})
        # Execute real model helpers, with a deterministic in-memory activation source.
        exec(compile(CELLS['model-helpers'], 'model-helpers', 'exec'), ns)
        exec("""def feature(row, kind='prompt'):
    return {str(layer): [float(row['response_harmful']) * 4 + layer, float(layer)] for layer in CONFIG['layers']}
""", ns)
        exec(compile(CELLS['fit'], 'synthetic-fit', 'exec'), ns)
        self.assertIn(ns['bundle']['selected_prompt_detector'], {'keyword', 'text', 'activation', 'fusion'})
        self.assertIn('monitors.json', ns['manifest']['stages'])
        exec(compile(CELLS['evaluate'], 'synthetic-evaluate', 'exec'), ns)
        self.assertEqual(len(ns['proxy']['predictions']), len(splits['final']))
        self.assertTrue((Path(self.temp.name) / 'final_predictions.jsonl').is_file())
        # Reconnect: cached monitor/evaluation must not call either feature extraction or score computation.
        ns['feature_matrix'] = lambda *_: self.fail('cached monitor was recomputed')
        exec(compile(CELLS['fit'], 'synthetic-fit-resume', 'exec'), ns)
        ns['all_scores'] = lambda *_: self.fail('cached proxy evaluation was recomputed')
        exec(compile(CELLS['evaluate'], 'synthetic-evaluate-resume', 'exec'), ns)
        # Seed a checkpointed paired audit, then execute the real audit cell through pending -> complete.
        ns['all_scores'] = globals().get('all_scores', ns['all_scores'])  # irrelevant after cached proxy load
        ns['AUDIT_BINDING'] = ns['digest']([ns['MONITOR_BINDING'], 'paired-greedy-audit', config['audit_n'],
                                            config['max_new_tokens'], config['review_rubric']])
        selected = splits['final'][:2]
        responses = []
        for row in selected:
            responses.extend([
                {'id': row['id'], 'condition': 'baseline', 'released': True, 'text': 'baseline'},
                {'id': row['id'], 'condition': 'prompt_gate', 'released': True, 'text': 'gated'},
            ])
        ns['save_stage']('audit_responses.json', {'ids': [r['id'] for r in selected], 'responses': responses,
                                                   'baseline_records': [], 'continuation_traces': []}, ns['AUDIT_BINDING'])
        exec(compile(CELLS['audit'], 'synthetic-audit-pending', 'exec'), ns)
        self.assertEqual(ns['audit_report']['status'], 'awaiting_independent_labels')
        review_path = Path(self.temp.name) / 'blinded_review.csv'
        with review_path.open(encoding='utf-8', newline='') as handle:
            review_rows = list(csv.DictReader(handle))
        with review_path.open('w', encoding='utf-8', newline='') as handle:
            fields = list(review_rows[0])
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
            for row in review_rows:
                writer.writerow(row | {'harmful_assistance': '0', 'task_success': '1', 'reviewer': 'test'})
        exec(compile(CELLS['audit'], 'synthetic-audit-complete', 'exec'), ns)
        self.assertEqual(ns['audit_report']['status'], 'complete_provisional')
        ns.update(stress=None, comparison=None, data={'records': [row for part in splits.values() for row in part]})
        exec(compile(CELLS['report'], 'synthetic-report', 'exec'), ns)
        report = json.loads((Path(self.temp.name) / 'final_report.json').read_text())
        self.assertEqual(report['status'], 'proxy_complete')
        self.assertEqual(report['generation_audit']['status'], 'complete_provisional')
        self.assertTrue((Path(self.temp.name) / 'REPORT.md').is_file())


    def test_helper_fingerprint_is_stable_across_python_hash_seeds(self):
        """Notebook identity is invariant to hash randomization and unrelated user helpers."""
        script = f"""
import csv, gc, hashlib, io, json, math, os, random, re, tempfile, time, warnings
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.exceptions import ConvergenceWarning
notebook = json.loads(Path({str(NOTEBOOK)!r}).read_text())
ns = globals()
for cell in notebook['cells']:
    if cell.get('id') in {{'storage', 'data-helpers', 'statistics', 'model-helpers', 'audit-helpers'}}:
        exec(compile(''.join(cell['source']), cell['id'], 'exec'), ns)
def user_scratch_function_not_part_of_notebook_contract():
    return {{'unordered', 'user', 'state'}}
print(helper_fingerprint())
"""
        def fingerprint(seed):
            environment = dict(os.environ, PYTHONHASHSEED=str(seed))
            return subprocess.check_output([sys.executable, '-c', script], text=True, env=environment).strip()
        self.assertEqual(fingerprint(1), fingerprint(987654))


    def test_identity_accepts_gpu_changes_as_observations_but_rejects_config_changes(self):
        """A Colab GPU swap must not invalidate artifacts; protocol/config changes must."""
        import importlib
        import types
        ns = namespace(self.temp.name)
        # helper_fingerprint is intentionally a complete fixed helper contract.
        exec(compile(CELLS['model-helpers'], 'model-helpers', 'exec'), ns)
        class Cuda:
            name = 'T4'
            def get_device_name(self, index): return self.name
            def get_device_capability(self, index): return (7, 5)
        fake_torch = types.SimpleNamespace(cuda=Cuda(), version=types.SimpleNamespace(cuda='12.4'))
        config = {'protocol': 'locked'}
        code = {'release': ns['IMPLEMENTATION_SHA256'], 'helpers': ns['helper_fingerprint']()}
        execution = {'packages': {name: 'test' for name in ['torch', 'transformers', 'datasets', 'huggingface_hub', 'accelerate', 'numpy', 'scikit-learn']},
                     'dtype': 'float16', 'attention': 'eager', 'python': list(sys.version_info[:3]), 'cuda': '12.4'}
        manifest = {'run_id': 'identity', 'config': config, 'execution': execution, 'code': code,
                    'dataset_revision': 'a' * 40, 'stages': {}}
        ns['atomic_json'](Path(self.temp.name) / 'manifest.json', manifest)
        (Path(self.temp.name) / 'dataset_candidates.json').write_text('{}', encoding='utf-8')
        ns.update(RUN_ID='identity', CONFIG=config, MODE='run', DTYPE='float16', torch=fake_torch,
                  importlib=types.SimpleNamespace(metadata=types.SimpleNamespace(version=lambda _:'test')),
                  sys=sys, re=__import__('re'), get_token=lambda: self.fail('token should not be requested'),
                  HfApi=lambda **_: self.fail('dataset lookup should not be requested'))
        exec(compile(CELLS['identity'], 'identity-first-gpu', 'exec'), ns)
        first_binding = ns['BASE_BINDING']
        fake_torch.cuda.name = 'L4'
        exec(compile(CELLS['identity'], 'identity-second-gpu', 'exec'), ns)
        self.assertEqual(ns['BASE_BINDING'], first_binding)
        persisted = ns['json_read'](Path(self.temp.name) / 'manifest.json')
        self.assertEqual([item['gpu'] for item in persisted['hardware_observations']], ['T4', 'L4'])
        ns['CONFIG'] = {'protocol': 'changed'}
        with self.assertRaisesRegex(RuntimeError, 'Configuration mismatch'):
            exec(compile(CELLS['identity'], 'identity-config-change', 'exec'), ns)


    def test_tiny_random_qwen_hooks_and_generation_contract_when_transformers_is_available(self):
        """Hook placement and EOS-free continuation content work on a real tiny decoder."""
        try:
            from transformers import GenerationConfig, Qwen2Config, Qwen2ForCausalLM
        except ImportError:
            self.skipTest('transformers/Qwen2 unavailable')
        import gc
        config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=4,
                             num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
                             eos_token_id=2, pad_token_id=0, bos_token_id=1)
        config._attn_implementation = 'eager'
        model = Qwen2ForCausalLM(config).eval().requires_grad_(False)
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs): return [1, 3, 4]
            def encode(self, text, **kwargs): return [5, 6]
            def decode(self, values, **kwargs): return str(values)
        ns = self.ns
        ns.update(torch=torch, gc=gc, model=model, tokenizer=Tokenizer(), DEVICE=torch.device('cpu'),
                  CONFIG={'layers': [0, 1], 'max_pair_tokens': 16, 'max_new_tokens': 2},
                  GENERATION=GenerationConfig(max_new_tokens=2, do_sample=False, eos_token_id=2, pad_token_id=0))
        exec(compile(CELLS['model-helpers'], 'tiny-qwen-model-helpers', 'exec'), ns)
        exec(compile(CELLS['audit-helpers'], 'tiny-qwen-audit-helpers', 'exec'), ns)
        captured = ns['extract_ids']([1, 3, 4], [0, 1])
        self.assertEqual(set(captured), {'0', '1'})
        self.assertTrue(all(len(vector) == 16 and np.isfinite(vector).all() for vector in captured.values()))
        baseline = ns['generate_baseline']({'id': 'tiny', 'prompt': 'hello', 'response': 'unused'})
        self.assertIn('content_token_ids', baseline)
        self.assertLessEqual(len(baseline['content_token_ids']), len(baseline['token_ids']))


if __name__ == '__main__':
    unittest.main()
