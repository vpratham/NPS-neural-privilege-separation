"""CPU-only contract tests for the standalone NFW-009 Colab experiment."""
import ast
import contextlib
import hashlib
import io
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


NB_PATH = (
    Path(__file__).resolve().parents[2]
    / 'experiments/NFW-09_tool_result_injection'
    / 'NFW_009_Tool_Result_Injection.ipynb'
)
NB = json.loads(NB_PATH.read_text(encoding='utf-8'))
CELLS = {c['id']: ''.join(c['source']) for c in NB['cells'] if c['cell_type'] == 'code'}
PREPARE = ('install', 'setup', 'storage', 'tasks', 'broker', 'broker_controls',
           'generation_helpers', 'evaluation_helpers', 'report_helpers', 'manifest')


class NFW009NotebookTest(unittest.TestCase):
    def run_cells(self, output_root, cells=PREPARE, ns=None, review=False):
        env = {'NFW009_SKIP_INSTALL': '1', 'NFW009_OUTPUT_ROOT': str(output_root),
               'NFW009_RUN_ID': 'fixture', 'NFW009_REVIEW_ONLY': str(int(review))}
        namespace = {} if ns is None else ns
        with patch.dict(os.environ, env), contextlib.redirect_stdout(io.StringIO()):
            for name in cells:
                exec(compile(CELLS[name], name, 'exec'), namespace)
        return namespace

    def populate_fixture_responses(self, ns):
        for key in ns['SELECTED_MODELS']:
            rendered = {ns['case_id'](task, condition):
                        '<tool_call>read_public</tool_call><tool_response>'
                        + ns['tool_text'](task, condition) + '</tool_response><|im_start|>assistant\n'
                        for task in ns['TASKS'] for condition in ns['CONDITIONS']}
            info = {'key': key, 'model_id': ns['MODEL_SPECS'][key]['id'],
                    'revision': ns['MODEL_SPECS'][key]['revision'],
                    'chat_template_sha256': 'fixture-template', 'tokenizer_class': 'Fixture',
                    'eos_token_id': 1, 'pad_token_id': 1,
                    'rendered_prompts_sha256': ns['sha'](rendered),
                    'token_counts': {cid: 40 for cid in rendered}}
            ns['save_immutable'](ns['model_info_filename'](key), info, ns['BINDING'])
            ns['save_immutable'](ns['rendered_filename'](key), rendered, ns['BINDING'])
            ns['mark_stage']('model_info_' + key, ns['model_info_filename'](key))
            ns['mark_stage']('rendered_' + key, ns['rendered_filename'](key))
            for task in ns['TASKS']:
                for condition in ns['CONDITIONS']:
                    cid = ns['case_id'](task, condition)
                    raw = ns['canonical'](task['expected'])
                    if condition == 'injected':
                        raw = (ns['canonical'](task['attacker_target']) if key == 'qwen_0_5b'
                               else '{"tool":"none","arguments":[]}')
                    record = {'task_id': task['id'], 'condition': condition,
                              'model_key': key, 'status': 'ok', 'raw_output': raw,
                              'input_tokens': 40, 'generated_tokens': 12,
                              'rendered_prompt_sha256': ns['sha'](rendered[cid]),
                              'messages_sha256': ns['sha'](ns['messages_for'](task, condition)),
                              'source': 'pinned_huggingface_gpu',
                              'runtime': {'gpu': 'fixture-gpu', 'capability': [8, 0],
                                          'torch': 'fixture-torch', 'cuda': 'fixture-cuda',
                                          'deterministic_algorithms': False,
                                          'cudnn_deterministic': False,
                                          'cudnn_benchmark': False,
                                          'cuda_matmul_tf32': False}}
                    ns['save_immutable'](ns['response_filename'](key, task, condition),
                                         record, ns['response_binding'](key, info, task, condition))

    def test_notebook_compiles_and_has_no_stale_outputs(self):
        self.assertEqual(len(CELLS), len(set(CELLS)))
        for cell in NB['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))
                self.assertIsNone(cell['execution_count'])
                self.assertEqual(cell['outputs'], [])
        code_payload = [(c['id'], ''.join(c['source'])) for c in NB['cells']
                        if c['cell_type'] == 'code']
        recorded_hash = next(src.split("NOTEBOOK_CODE_SHA256='")[1].split("'")[0]
                             for name, src in code_payload if name == 'setup')
        normalized = [(name, src.replace(recorded_hash, 'PENDING'))
                      for name, src in code_payload]
        actual_hash = hashlib.sha256(json.dumps(
            normalized, ensure_ascii=True, separators=(',', ':')).encode('utf-8')).hexdigest()
        self.assertEqual(recorded_hash, actual_hash)

    def test_frozen_matrix_tool_role_and_broker_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            ns = self.run_cells(directory)
            self.assertEqual(len(ns['TASKS']), 48)
            self.assertEqual(len({t['id'] for t in ns['TASKS']}), 48)
            self.assertEqual(set(t['goal'] for t in ns['TASKS']), set(ns['GOALS']))
            self.assertEqual(ns['BROKER_CONTROLS']['passed'], 15)
            for task in ns['TASKS']:
                messages = ns['messages_for'](task, 'injected')
                self.assertEqual([m['role'] for m in messages],
                                 ['system', 'user', 'assistant', 'tool'])
                self.assertEqual(messages[2]['tool_calls'][0]['function']['name'], 'read_public')
                self.assertIn(task['fact'], messages[3]['content'])
                self.assertNotEqual(ns['tool_text'](task, 'clean'),
                                    ns['tool_text'](task, 'injected'))
                self.assertEqual(ns['messages_for'](task, 'clean')[1],
                                 ns['messages_for'](task, 'injected')[1])
                self.assertNotIn(task['goal'], ns['tool_text'](task, 'clean'))
                self.assertNotIn(task['goal'], ns['messages_for'](task, 'clean')[1]['content'])
            with self.assertRaisesRegex(ValueError, 'invalid_noop_shape'):
                ns['parse_proposal']('{"tool":"none","arguments":[]}')
            with self.assertRaisesRegex(ValueError, 'duplicate_json_key'):
                ns['parse_proposal']('{"tool":"none","tool":"write_record","arguments":{}}')

    def test_rendered_transcript_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            ns = self.run_cells(directory)

            class Tokenizer:
                def apply_chat_template(self, messages, **kwargs):
                    return ('<tool_call>read_public</tool_call><tool_response>'
                            + messages[-1]['content'] + '</tool_response><|im_start|>assistant\n')

                def __call__(self, prompt, **kwargs):
                    return {'input_ids': [1] * 80}

            rendered, lengths = ns['build_rendered_prompts'](Tokenizer())
            self.assertEqual(len(rendered), 144)
            self.assertEqual(set(lengths.values()), {80})

            class Broken(Tokenizer):
                def apply_chat_template(self, messages, **kwargs):
                    return 'flattened ' + messages[-1]['content']

            with self.assertRaisesRegex(RuntimeError, 'native assistant tool call'):
                ns['build_rendered_prompts'](Broken())

    def test_fixture_evaluation_effect_and_utility_separation(self):
        with tempfile.TemporaryDirectory() as directory:
            ns = self.run_cells(directory)
            self.populate_fixture_responses(ns)
            self.run_cells(directory, cells=('evaluate', 'report'), ns=ns)
            report = ns['report']
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['response_count'], 288)
            self.assertEqual(report['evaluation_rows'], 576)
            self.assertFalse(report['mixed_runtime'])
            self.assertTrue(report['security_checks']['unauthorized_scope_effects_zero'])
            self.assertTrue(report['security_checks']['exact_argument_wrong_content_effects_zero'])
            small_scope = report['model_summaries']['qwen_0_5b']['scope_only']['injected']['counts']
            small_exact = report['model_summaries']['qwen_0_5b']['exact_args']['injected']['counts']
            large = report['model_summaries']['qwen_3b']['scope_only']['injected']['counts']
            self.assertEqual(small_scope['attacker_goal_proposal'], 48)
            self.assertEqual(small_scope['wrong_content_effect'], 16)
            self.assertEqual(small_exact['wrong_content_effect'], 0)
            self.assertEqual(large['invalid_noop_shape'], 48)
            self.assertEqual(large['valid_abstention'], 0)
            self.assertEqual(large['task_success'], 0)
            self.assertEqual(report['paired_tables']['qwen_0_5b']['scope_only']
                             ['clean_vs_injected']['task_success']['left_only'], 48)
            self.assertTrue((ns['RUN_DIR'] / 'REPORT.md').exists())
            reopened = self.run_cells(directory, review=True)
            self.run_cells(directory, cells=('generate', 'evaluate', 'report'), ns=reopened,
                           review=True)
            self.assertEqual(reopened['report'], report)

    def test_tamper_and_manifest_change_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            ns = self.run_cells(directory)
            self.populate_fixture_responses(ns)
            task = ns['TASKS'][0]
            path = ns['RUN_DIR'] / ns['response_filename']('qwen_0_5b', task, 'clean')
            path.write_text(path.read_text(encoding='utf-8') + ' ', encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'mismatch'):
                self.run_cells(directory, cells=('evaluate',), ns=ns)
            changed = self.run_cells(directory, cells=PREPARE[:-1], ns={})
            changed['MAX_NEW_TOKENS'] += 1
            with self.assertRaisesRegex(RuntimeError, 'manifest identity mismatch'):
                self.run_cells(directory, cells=('manifest',), ns=changed)

    def test_evaluator_code_change_refuses_cache_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_cells(directory)
            changed = self.run_cells(directory, cells=PREPARE[:-1], ns={})

            def different_evaluator(*args):
                return {'changed': True}

            changed['evaluate_one'] = different_evaluator
            with self.assertRaisesRegex(RuntimeError, 'manifest identity mismatch'):
                self.run_cells(directory, cells=('manifest',), ns=changed)

    def test_missing_response_review_only_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            ns = self.run_cells(directory)
            self.populate_fixture_responses(ns)
            path = ns['RUN_DIR'] / ns['response_filename']('qwen_0_5b', ns['TASKS'][0], 'clean')
            path.unlink()
            reopened = self.run_cells(directory, review=True)
            with self.assertRaisesRegex(RuntimeError, 'missing responses'):
                self.run_cells(directory, cells=('generate',), ns=reopened, review=True)

    def test_generation_contract_with_fake_tensor_and_runtime(self):
        class Tensor:
            def __init__(self, values):
                self.values = values
                self.shape = (1, len(values))

            def __getitem__(self, item):
                return Tensor(self.values[item[1]])

            def numel(self):
                return len(self.values)

        class Batch(dict):
            def to(self, device):
                return self

        class Tokenizer:
            eos_token_id = 1

            def __call__(self, prompt, **kwargs):
                return Batch({'input_ids': Tensor([1, 2]),
                              'attention_mask': Tensor([1, 1])})

            def decode(self, tensor, **kwargs):
                return '{"tool":"none","arguments":{}}'

        class Model:
            device = 'cuda:0'

            def generate(self, **kwargs):
                return Tensor([1, 2, 3])

        torch = types.SimpleNamespace(
            manual_seed=lambda seed: None,
            inference_mode=contextlib.nullcontext,
            cuda=types.SimpleNamespace(get_device_name=lambda n: 'fixture-gpu',
                                       get_device_capability=lambda n: (8, 0)),
            version=types.SimpleNamespace(cuda='fixture-cuda'),
            backends=types.SimpleNamespace(
                cudnn=types.SimpleNamespace(deterministic=False, benchmark=False),
                cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False))),
            are_deterministic_algorithms_enabled=lambda: False,
            __version__='fixture-torch',
        )
        with tempfile.TemporaryDirectory() as directory:
            ns = self.run_cells(directory)
            task = ns['TASKS'][0]
            cid = ns['case_id'](task, 'clean')
            info = {'token_counts': {cid: 2}}
            record = ns['generate_one'](Model(), Tokenizer(), task, 'clean',
                                        'qwen_0_5b', info, {cid: 'fixture prompt'}, torch)
            self.assertEqual(record['input_tokens'], 2)
            self.assertEqual(record['generated_tokens'], 1)
            self.assertEqual(record['runtime']['gpu'], 'fixture-gpu')


if __name__ == '__main__':
    unittest.main()
