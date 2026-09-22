"""CPU-only contract tests for the NFW-008 Colab notebook; no model weights are loaded."""
import ast
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

NB_PATH = (
    Path(__file__).resolve().parents[2]
    / 'experiments/NFW-08_live_model_broker_evaluation'
    / 'NFW_008_Live_Model_Capability_Broker.ipynb'
)
NB = json.loads(NB_PATH.read_text(encoding='utf-8'))
CELLS = {c['id']: ''.join(c['source']) for c in NB['cells'] if c['cell_type'] == 'code'}
PREPARE = ('install', 'setup', 'storage', 'tasks', 'broker', 'generation_helpers',
           'evaluation_helpers', 'manifest', 'broker_controls')


class NFW008NotebookTest(unittest.TestCase):
    def run_cells(self, output_root, run_id='fixture', cells=PREPARE, ns=None):
        env = {'NFW008_SKIP_INSTALL': '1', 'NFW008_OUTPUT_ROOT': str(output_root),
               'NFW008_RUN_ID': run_id}
        namespace = {'__name__': '__main__'} if ns is None else ns
        with patch.dict(os.environ, env), contextlib.redirect_stdout(io.StringIO()):
            for name in cells:
                exec(compile(CELLS[name], name, 'exec'), namespace)
        return namespace

    def populate_fixture_responses(self, ns):
        for key in ns['SELECTED_MODELS']:
            spec = ns['MODEL_SPECS'][key]
            info = {'key': key, 'model_id': spec['id'], 'revision': spec['revision'],
                    'chat_template_sha256': 'fixture-template', 'tokenizer_class': 'Fixture',
                    'eos_token_id': 1, 'pad_token_id': 1}
            name = f'model_info/{key}.json'
            ns['save_immutable'](name, info, ns['BINDING'])
            ns['mark_stage']('model_info_' + key, name)
            for task in ns['TASKS']:
                raw = ns['canonical'](task['expected'])
                if key == 'qwen_0_5b' and task['id'] == 'benign_write_00':
                    raw = ns['canonical']({'tool': 'write_record',
                                           'arguments': {'resource': 'notes', 'value': 'wrong'}})
                if key == 'qwen_3b' and task['id'] == 'injected_read_00':
                    raw = ns['canonical']({'tool': 'write_record',
                                           'arguments': {'resource': 'protected', 'value': 'leak'}})
                record = {'task_id': task['id'], 'model_key': key, 'status': 'ok',
                          'raw_output': raw, 'input_tokens': 40, 'generated_tokens': 12,
                          'prompt_sha256': ns['sha'](ns['messages_for'](task)),
                          'source': 'pinned_huggingface_gpu'}
                ns['save_immutable'](ns['response_filename'](key, task), record,
                                     ns['response_binding'](key, info, task))

    def test_notebook_compiles_without_saved_outputs(self):
        for cell in NB['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])

    def test_task_labels_and_broker_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            ns = self.run_cells(directory)
            self.assertEqual(len(ns['TASKS']), 48)
            self.assertEqual(len({t['id'] for t in ns['TASKS']}), 48)
            self.assertEqual(ns['BROKER_CONTROLS']['passed'], 7)
            with self.assertRaisesRegex(ValueError, 'wire_schema'):
                ns['parse_proposal']('{"tool":"read_public","arguments":{"resource":"public"},"capability":"admin"}')
            with self.assertRaisesRegex(ValueError, 'duplicate_json_key'):
                ns['parse_proposal']('{"tool":"read_public","tool":"write_record","arguments":{"resource":"public"}}')

    def test_paired_evaluation_and_argument_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            ns = self.run_cells(directory)
            self.populate_fixture_responses(ns)
            self.run_cells(directory, cells=('evaluate', 'report'), ns=ns)
            report = ns['report']
            self.assertEqual(report['status'], 'complete')
            small_scope = report['model_summaries']['qwen_0_5b']['scope_only']
            small_exact = report['model_summaries']['qwen_0_5b']['exact_args']
            self.assertEqual(small_scope['wrong_content_effects'], 1)
            self.assertEqual(small_exact['wrong_content_effects'], 0)
            self.assertEqual(report['model_summaries']['qwen_3b']['scope_only']['unauthorized_scope_effects'], 0)
            self.assertEqual(report['paired_model_comparison']['scope_only']['second_only'], 1)
            self.assertEqual(report['paired_model_comparison']['scope_only']['first_only'], 1)
            self.assertEqual(report['model_summaries']['qwen_3b']['scope_only']['by_family']['injected_read']['protected_proposals'], 1)
            self.assertEqual(report['model_summaries']['qwen_3b']['scope_only']['injection_vs_clean_pairs']['clean_only'], 1)
            # Reopening validates and reuses immutable output rather than regenerating.
            reopened = self.run_cells(directory)
            self.run_cells(directory, cells=('evaluate', 'report'), ns=reopened)
            self.assertEqual(reopened['report'], report)

    def test_changed_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            ns = self.run_cells(directory)
            self.populate_fixture_responses(ns)
            self.run_cells(directory, cells=('evaluate',), ns=ns)
            path = Path(directory) / 'fixture/responses/qwen_0_5b/benign_read_00.json'
            path.write_text(path.read_text(encoding='utf-8') + ' ', encoding='utf-8')
            reopened = self.run_cells(directory)
            with self.assertRaisesRegex(RuntimeError, 'checksum|mismatch'):
                self.run_cells(directory, cells=('evaluate',), ns=reopened)

    def test_missing_checkpoint_refuses_review(self):
        with tempfile.TemporaryDirectory() as directory:
            ns = self.run_cells(directory)
            self.populate_fixture_responses(ns)
            path = Path(directory) / 'fixture/responses/qwen_0_5b/benign_read_00.json'
            path.unlink()
            with self.assertRaisesRegex(RuntimeError, 'Missing response'):
                self.run_cells(directory, cells=('evaluate',), ns=ns)

    def test_review_only_skips_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            ns = self.run_cells(directory)
            self.populate_fixture_responses(ns)
            with patch.dict(os.environ, {'NFW008_REVIEW_ONLY': '1'}):
                reopened = self.run_cells(directory)
                self.assertTrue(reopened['REVIEW_ONLY'])
                self.run_cells(directory, cells=('generate', 'evaluate', 'report'), ns=reopened)
            self.assertEqual(reopened['report']['status'], 'complete')

    def test_generation_contract_with_fake_tensor(self):
        class Tensor:
            def __init__(self, values):
                self.values = values
                self.shape = (1, len(values))

            def to(self, device):
                return self

            def __getitem__(self, item):
                return Tensor(self.values[item[1]])

            def numel(self):
                return len(self.values)

        class Tokenizer:
            eos_token_id = 1

            def apply_chat_template(self, messages, **kwargs):
                return Tensor([2, 3, 4])

            def decode(self, tensor, **kwargs):
                return '{"tool":"none","arguments":{}}'

        class Model:
            device = 'cuda:0'

            def generate(self, **kwargs):
                return Tensor([2, 3, 4, 5, 6])

        class Torch:
            @staticmethod
            def manual_seed(seed):
                pass

            @staticmethod
            def inference_mode():
                return contextlib.nullcontext()

            @staticmethod
            def ones_like(tensor):
                return tensor

        with tempfile.TemporaryDirectory() as directory:
            ns = self.run_cells(directory)
            record = ns['generate_one'](Model(), Tokenizer(), ns['TASKS'][0],
                                        'qwen_0_5b', Torch())
            self.assertEqual(record['status'], 'ok')
            self.assertEqual(record['input_tokens'], 3)
            self.assertEqual(record['generated_tokens'], 2)
            self.assertEqual(record['source'], 'pinned_huggingface_gpu')


if __name__ == '__main__':
    unittest.main()
