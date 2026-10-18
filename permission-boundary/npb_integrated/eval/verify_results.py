"""Verify frozen pilot artifacts and recompute scores without model calls."""
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.score import score_record, summarize


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify(directory):
    config = json.loads((directory / 'config.json').read_text())
    cases_path = Path(config['cases'])
    if not cases_path.exists():
        cases_path = ROOT / ('results/replay_cases.jsonl' if directory.name == 'replay' else 'eval/cases.jsonl')
    assert hashlib.sha256(cases_path.read_bytes()).hexdigest() == config['cases_sha256']
    for name, expected in config['code'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    cases = read_jsonl(cases_path)[:config['case_count']]
    comparison = json.loads((directory / 'comparison.json').read_text())
    by_id = {case['id']: case for case in cases}
    seen_questions = {}
    for case in cases:
        previous = seen_questions.setdefault(case['pair_id'], case['question'])
        assert previous == case['question'], case['pair_id']
    results = {}
    for system in ['baseline', 'defended']:
        rows = read_jsonl(directory / (system + '_results.jsonl'))
        assert len(rows) == len(cases)
        assert {r['case_id'] for r in rows} == set(by_id)
        for row in rows:
            assert row['case'] == by_id[row['case_id']]
            assert row['scores'] == score_record(row['case'], row['result'])
            generation = row['result'].get('generation')
            if generation:
                assert generation['gpu_status']['size_vram'] == 0
                assert generation['gpu_status']['cpu_only_verified'] is True
                assert generation['input_tokens'] > 0 and generation['output_tokens'] > 0
                assert generation['messages']
            for span in row['result']['supporting_spans']:
                assert row['case']['document'][span['start']:span['end']] == span['text']
        assert comparison[system] == summarize(cases, rows)
        results[system] = rows
    calls = sum(r['result'].get('generation') is not None for rows in results.values() for r in rows)
    assert comparison['model_calls_attempted'] == calls <= 2 * len(cases)
    assert config['provider']['options']['num_gpu'] == 0
    assert config['provider']['options']['temperature'] == 0
    return results, calls


def main():
    rows, initial = verify(ROOT / 'results')
    repeats, extra = verify(ROOT / 'results/replay')
    config = json.loads((ROOT / 'results/config.json').read_text())
    replay_config = json.loads((ROOT / 'results/replay/config.json').read_text())
    assert config['provider'] == replay_config['provider']
    assert config['code'] == replay_config['code']
    originals = {r['case_id']: r for r in rows['baseline']}
    for system in repeats:
        first = {r['case_id']: r for r in rows[system]}
        for row in repeats[system]:
            assert row['case'] == originals[row['case_id']]['case']
            assert row['result']['answer'] == first[row['case_id']]['result']['answer']
    assert initial + extra <= 80
    assert len(rows['baseline']) + len(repeats['baseline']) <= 40
    failed = json.loads((ROOT / 'results/development_failures.json').read_text())
    original_source = (ROOT / 'results/initial_extraction.py.txt').read_text()
    assert hashlib.sha256(original_source.encode()).hexdigest() == failed['parser_sha256']
    archive = {}
    exec(compile(original_source, 'initial_extraction.py.txt', 'exec'), archive)
    for record in failed['records']:
        case = record['case']
        assert archive['extract'](case['question'], case['document']) == record['actual_response']['extraction']
    by_case = {name: {r['case_id']: r for r in values} for name, values in rows.items()}
    ratios = [by_case['defended'][key]['runtime_seconds'] / base['runtime_seconds']
              for key, base in by_case['baseline'].items()]
    generated_ratios = [by_case['defended'][key]['runtime_seconds'] / base['runtime_seconds']
                        for key, base in by_case['baseline'].items()
                        if by_case['defended'][key]['result'].get('generation')]
    print(json.dumps({'pilot_inputs': len(rows['baseline']), 'repeat_inputs': len(repeats['baseline']),
                      'pilot_calls': initial, 'repeat_calls': extra, 'total_calls': initial + extra,
                      'source_and_score_checks': 'passed', 'repeat_answer_checks': 'passed',
                      'paired_latency_ratio_median': statistics.median(ratios),
                      'paired_latency_ratio_median_both_generated': statistics.median(generated_ratios)}, indent=2))


if __name__ == '__main__':
    main()
