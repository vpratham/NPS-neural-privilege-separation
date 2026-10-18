"""Verify held-out results, source freezes and scores; no inference or writes."""
import json
import statistics
from pathlib import Path

from run_eval import HERE, REPO, read_jsonl, verify_freeze
from score import score_record, summarize
from reproduce_failure import reproduce
from npb_integrated.src.baseline import PROMPT_TEMPLATE
from npb_integrated.src.defended import messages_for
from npb_integrated.src.extraction import extract


def main():
    freeze, protocol = verify_freeze()
    cases = read_jsonl(HERE / 'cases.jsonl')
    truths = read_jsonl(HERE / 'ground_truth.jsonl')
    by_case, by_truth = {c['id']: c for c in cases}, {t['case_id']: t for t in truths}
    scores = json.loads((HERE / 'scores.json').read_text())
    execution = json.loads((HERE / 'execution_config.json').read_text())
    assert execution['provider']['model_digest'] == freeze['model']['model_digest']
    assert execution['provider']['options'] == freeze['model']['options']
    all_rows, phases, missing, calls, repeated_matches = [], {}, [], 0, 0
    expected_repeats = set(protocol['repeat_case_ids'])
    originals = {}
    for phase, prefix in [('primary', ''), ('repeat', 'repeat_')]:
        phase_rows = []
        for system in ['baseline', 'defended']:
            rows = read_jsonl(HERE / (prefix + system + '_results.jsonl'))
            expected = set(by_case) if phase == 'primary' else expected_repeats
            assert len(rows) == len(expected) and {r['case_id'] for r in rows} == expected
            for row in rows:
                case = by_case[row['case_id']]
                assert row['question'] == case['question'] and row['document'] == case['document']
                assert row['scores'] == score_record(case, by_truth[case['id']], row['result'], system)
                calls += row['model_calls_attempted']
                gen = row['result'].get('generation')
                if gen:
                    assert row['model_calls_attempted'] == 1
                    assert gen['gpu_status']['cpu_only_verified'] and gen['gpu_status']['size_vram'] == 0
                    assert gen['input_tokens'] > 0 and gen['output_tokens'] > 0 and gen['messages']
                    assert gen['input_tokens'] <= 4096 and gen['output_tokens'] <= 128
                    expected_messages = ([{'role': 'user', 'content': PROMPT_TEMPLATE.format(
                        question=case['question'], document=case['document'])}] if system == 'baseline'
                        # Recompute insertion order; sorted result JSON does not preserve prompt serialization order.
                        else messages_for(case['question'], case['document'], extract(case['question'], case['document'])['facts']))
                    assert gen['messages'] == expected_messages
                else:
                    assert row['model_calls_attempted'] == 0
                if phase == 'primary':
                    originals[system, case['id']] = row
                else:
                    original = originals[system, case['id']]['result']
                    same = all(row['result'].get(k) == original.get(k)
                               for k in ('status', 'answer', 'supporting_spans', 'reason'))
                    repeated_matches += same
            phase_rows.extend(rows)
        assert scores['primary' if phase == 'primary' else 'repeats'] == summarize(cases, truths, phase_rows)
        phases[phase] = phase_rows
        all_rows.extend(phase_rows)
    assert calls == scores['model_calls_total'] <= 120
    assert len(phases['primary']) == 96 and len(phases['repeat']) == 16
    failed = [r for r in phases['primary'] if r['system'] == 'defended'
              and r['scores']['correct_task_completion'] is not True]
    interventions = []
    for row in failed:
        witness = reproduce(by_case[row['case_id']], row)
        assert witness['reproduced']
        if witness['attacker_span_removal'] is not None:
            interventions.append({'case_id': row['case_id'],
                'attacked_status': row['result']['status'],
                'clean_preflight_status': witness['attacker_span_removal']['preflight']['status'],
                'scope': 'preflight only, no clean model answer generated'})
    baseline = {r['case_id']: r for r in phases['primary'] if r['system'] == 'baseline'}
    defended = [r for r in phases['primary'] if r['system'] == 'defended']
    paired_called = [r['runtime_seconds'] / baseline[r['case_id']]['runtime_seconds']
                     for r in defended if r['model_calls_attempted']]
    print(json.dumps({'reference_commit': freeze['reference_commit'],
        'reference_files_verified': len(freeze['reference_files']),
        'runtime_dependencies_verified': len(freeze['runtime_dependencies']),
        'protocol_files_verified': len(protocol['files']), 'system_requests': len(all_rows),
        'model_calls': calls, 'primary_calls': sum(r['model_calls_attempted'] for r in phases['primary']),
        'repeat_calls': sum(r['model_calls_attempted'] for r in phases['repeat']),
        'skipped_generation_requests': len(all_rows) - calls,
        'repeat_status_answer_span_matches': repeated_matches,
        'defense_failures_reproduced_without_inference': len(failed),
        'attacker_span_interventions': interventions,
        'paired_latency_ratio_median_both_generated': statistics.median(paired_called),
        'result_checks': 'passed'}, indent=2))


if __name__ == '__main__':
    main()
