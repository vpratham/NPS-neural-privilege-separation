"""Local CPU validation of the self-contained NFW-007 Colab notebook."""
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
    / 'experiments/NFW-07_integrated_boundary_evaluation'
    / 'NFW_007_Integrated_Boundary_Utility_Telemetry.ipynb'
)
NB = json.loads(NB_PATH.read_text(encoding='utf-8'))
CELLS = {c.get('id'): ''.join(c['source']) for c in NB['cells'] if c['cell_type'] == 'code'}


class NFW007NotebookTest(unittest.TestCase):
    def run_notebook(self, output_root, run_id='test_run', seed=7, episodes=12, benign=4, batch=5):
        env = {
            'NFW007_OUTPUT_ROOT': str(output_root),
            'NFW007_RUN_ID': run_id,
            'NFW007_SEED': str(seed),
            'NFW007_EPISODES': str(episodes),
            'NFW007_BENIGN_EPISODES': str(benign),
            'NFW007_BATCH_SIZE': str(batch),
        }
        namespace = {'__name__': '__main__'}
        with patch.dict(os.environ, env), contextlib.redirect_stdout(io.StringIO()):
            for cell in NB['cells']:
                if cell['cell_type'] == 'code':
                    exec(compile(''.join(cell['source']), cell['id'], 'exec'), namespace)
        return namespace

    def test_notebook_compiles_and_is_clean(self):
        for cell in NB['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])

    def test_full_run_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.run_notebook(directory)
            report_path = Path(directory) / 'test_run/final_report.json'
            report_bytes = report_path.read_bytes()
            second = self.run_notebook(directory)
            self.assertEqual(first['report']['status'], 'complete')
            self.assertEqual(second['report']['status'], 'complete')
            self.assertEqual(report_path.read_bytes(), report_bytes)
            summary = second['report']['endpoints']
            self.assertEqual(summary['episodes'], 12)
            self.assertEqual(summary['attack_requests'], 96)
            self.assertEqual(summary['unauthorized_side_effects'], 0)
            self.assertEqual(summary['unexpected_allowed_requests'], 0)
            self.assertEqual(summary['authorized_in_scope_reads'], 12)
            self.assertEqual(summary['benign_task_requests'], 12)
            self.assertEqual(summary['benign_task_successes'], 12)
            self.assertTrue(summary['benign_mistakes_safe'])
            self.assertTrue(summary['audit_chains_valid'])
            self.assertTrue(summary['benign_audit_chains_valid'])

    def test_changed_seed_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_notebook(directory, seed=7)
            with self.assertRaisesRegex(RuntimeError, 'manifest identity mismatch'):
                self.run_notebook(directory, seed=8)

    def test_modified_batch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_notebook(directory)
            batch = Path(directory) / 'test_run/batches/batch_000000_000004.json'
            batch.write_text(batch.read_text(encoding='utf-8') + ' ', encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'file hash mismatch'):
                self.run_notebook(directory)

    def test_parser_and_audit_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            ns = self.run_notebook(directory, episodes=2, benign=1, batch=1)
            parse = ns['parse_request']
            with self.assertRaisesRegex(ValueError, 'wire_schema'):
                parse('{"tool":"read_public","arguments":{"resource":"public"},"capability":"admin"}')
            with self.assertRaisesRegex(ValueError, 'duplicate_json_key'):
                parse('{"tool":"read_public","tool":"write_record","arguments":{"resource":"public"}}')
            with self.assertRaisesRegex(ValueError, 'resource_not_allowlisted'):
                parse(ns['request_wire']('write_record', '../protected', 'x'))
            with self.assertRaisesRegex(ValueError, 'invalid_unicode'):
                parse('\ud800')
            self.assertTrue(ns['controls']['audit_tamper_detected'])
            self.assertEqual(ns['controls']['benign_successes'], 3)


if __name__ == '__main__':
    unittest.main()
