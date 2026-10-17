"""Verify saved evidence and recompute descriptive counts; no inference or writes."""
import hashlib
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def pointer(value, path):
    for key in path.strip('/').split('/') if path else []:
        value = value[int(key)] if isinstance(value, list) else value[key.replace('~1', '/').replace('~0', '~')]
    return value


def main():
    frozen = read(HERE / 'evidence/protocol_freeze.json')
    for name, expected in frozen['files'].items():
        assert digest(HERE / name) == expected, name
    for row in read(HERE / 'evidence/local_source_manifest.json'):
        assert digest(ROOT / row['path']) == row['sha256'], row['path']
    retrievals = read(HERE / 'evidence/tab/retrievals.json')
    for row in retrievals:
        assert 'error' not in row, row
        assert digest(HERE / 'evidence/tab' / row['path']) == row['sha256'], row['path']

    rows = [json.loads(line) for line in (HERE / 'ANNOTATED_RECORDS.jsonl').read_text().splitlines()]
    primary = read(HERE / 'evidence/sample_plan.json')
    secondary = read(HERE / 'evidence/secondary_sample_plan.json')
    assert {r['case_id'] for r in rows if r['dataset'] == 'TAB-author-sample'} == set(primary['trial_ids'])
    expected_local = {f'NPB/{case}/{condition}/{arm}' for case in secondary['case_ids']
                      for condition in secondary['conditions'] for arm in secondary['arms']}
    assert {r['record_id'] for r in rows if r['dataset'] == 'NPB-BIPIA-local'} == expected_local
    assert len(rows) == len({r['record_id'] for r in rows}) == 35

    cache = {}
    def data(name):
        if name not in cache:
            cache[name] = read(ROOT / name)
        return cache[name]

    raw_local = data(secondary['source_file'])
    selected = read(HERE / 'evidence/local_selected_records.json')
    assert len(selected) == len(expected_local)
    assert {'NPB/' + r['record_id'] for r in selected} == expected_local
    for item in selected:
        original = raw_local['rows'][item['raw_row_index']]
        assert original['id'] == item['case_id']
        assert pointer(raw_local, item['raw_pointer']) == item['original_response']
        for field, hash_field in [('clean_source', 'source_context_sha256'),
                                  ('trusted_request', 'request_sha256'),
                                  ('reference', 'reference_sha256')]:
            assert hashlib.sha256(item[field].encode()).hexdigest() == original[hash_field]
        assert hashlib.sha256(item['trusted_policy'].encode()).hexdigest() == raw_local['policy_sha256']
        if item['condition'] != 'clean':
            assert hashlib.sha256(item['designated_attack'].encode()).hexdigest() == original['attack_sha256']
        serialized = json.dumps(tuple(item['documents'].items()), sort_keys=True,
                                separators=(',', ':'), allow_nan=False).encode()
        assert hashlib.sha256(serialized).hexdigest() == item['original_response']['documents_sha256']

    for row in rows:
        original_path = ROOT / row['original_record_file']
        assert digest(original_path) == row['original_record_sha256']
        for e in row['source_evidence'] + row['model_response_evidence']:
            assert digest(ROOT / e['file']) == e['file_sha256'], e['file']
            text = pointer(data(e['file']), e['pointer'])
            assert text[e['start']:e['end']] == e['quote'], row['record_id']
        if row['dataset'] == 'NPB-BIPIA-local':
            original = pointer(data(row['original_record_file']), row['original_record_pointer'])
            assert row['original_scores'] == {k: original[k] for k in ['reference_exact', 'reference_contained']}
            assert original['error_type'] is None and original['ended_with_eos'] is True
            assert re.fullmatch(r'\$\d+(?:,\d{3})*\.\d{2}', original['output'])
            assert row['outcomes']['C_unauthorized_redirection']['label'] == (
                'NOT APPLICABLE' if row['condition'] == 'clean' else 'NO')
        else:
            trial = original_path.parent.parent
            result = read(trial / 'result.json') if (trial / 'result.json').exists() else None
            verifier = read(trial / 'verifier/tab_result.json') if (trial / 'verifier/tab_result.json').exists() else None
            assert row['original_scores'] == {'result': result, 'tab_result': verifier}
            if row['outcomes']['C_unauthorized_redirection']['label'] == 'YES':
                assert verifier['injection']['executed'] is True
                source_pointer = row['source_evidence'][1]['pointer']
                action_pointer = row['model_response_evidence'][0]['pointer']
                assert int(source_pointer.split('/')[2]) < int(action_pointer.split('/')[2])
                assert '/tool_calls/' in action_pointer

    c_label = lambda r: r['outcomes']['C_unauthorized_redirection']['label']
    identified = [r for r in rows if c_label(r) in ('YES', 'NO')]
    groups = defaultdict(list)
    for row in identified:
        # Task and condition stay fixed; distinct tasks and score definitions never pool.
        key = (row['dataset'], row['case_id'], row['condition'], json.dumps(row['original_scores'], sort_keys=True))
        groups[key].append(row)
    pairs = [pair for group in groups.values() for pair in itertools.combinations(group, 2)]
    conflating = [(a['record_id'], b['record_id']) for a, b in pairs if c_label(a) != c_label(b)]
    scored_tab = [r for r in rows if r['dataset'] == 'TAB-author-sample' and r['score_available']]
    scalar_pairs = [(a['record_id'], b['record_id']) for a, b in itertools.combinations(scored_tab, 2)
                    if a['diagnostics']['historical_reward'] == b['diagnostics']['historical_reward']
                    and a['diagnostics']['historical_distractor_executed'] != b['diagnostics']['historical_distractor_executed']]
    local = [r for r in rows if r['dataset'] == 'NPB-BIPIA-local']
    output_groups = defaultdict(set)
    for row in local:
        output_groups[row['case_id']].add(row['diagnostics']['output_sha256'])
    result = {
        'selected_record_units': len(rows),
        'units_with_saved_model_output': sum(r['response_available'] for r in rows),
        'units_with_saved_output_and_original_scores': sum(r['response_available'] and r['score_available'] for r in rows),
        'identified_applicable_C_records': len(identified),
        'identified_C_states': sorted({c_label(r) for r in identified}),
        'outcome_label_counts': {axis: dict(Counter(r['outcomes'][axis]['label'] for r in rows)) for axis in rows[0]['outcomes']},
        'records_with_any_indeterminate_label': sum(any(v['label'] == 'INDETERMINATE' for v in r['outcomes'].values()) for r in rows),
        'same_task_same_condition_equal_score_pairs_with_identified_C': len(pairs),
        'identified_C_conflation_pairs': conflating,
        'identified_C_conflation_participant_count': len({i for pair in conflating for i in pair}),
        'full_three_way_applicable_records': sum(all(v['label'] in ('YES', 'NO')
                                                    for v in r['outcomes'].values()) for r in rows),
        'TAB_scalar_reward_different_existing_execution_flag_pairs': scalar_pairs,
        'TAB_scalar_reward_collision_participants': len({i for pair in scalar_pairs for i in pair}),
        'local_unique_output_counts_by_case': {k: len(v) for k, v in output_groups.items()},
        'local_reference_mismatches': sum(r['diagnostics']['reference_mismatch'] for r in local),
        'observed_defense_ranking_change': False,
        'ranking_scope': 'Three NPB arms tie on the selected cases; external TAB trials do not pair defenses on the same tasks.',
        'decision': 'INCONCLUSIVE',
    }
    assert len(identified) == 22 and len(pairs) == 18 and not conflating
    assert len(scalar_pairs) == 3 and result['TAB_scalar_reward_collision_participants'] == 4
    assert result['units_with_saved_model_output'] == 34 and result['units_with_saved_output_and_original_scores'] == 32
    assert all(v == 1 for v in result['local_unique_output_counts_by_case'].values())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
