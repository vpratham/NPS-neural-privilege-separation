"""Held-out harness importing the frozen assistants without modifying them."""
import argparse
import datetime
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOUNDARY = HERE.parent
REPO = BOUNDARY.parent
sys.path[:0] = [str(BOUNDARY), str(REPO), str(HERE)]

from npb_integrated.src.authorization import Authorization
from npb_integrated.src.baseline import BaselineAssistant
from npb_integrated.src.defended import DefendedAssistant
from npb_integrated.src.providers import OllamaProvider
from score import score_record, summarize


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify_freeze():
    freeze = json.loads((HERE / 'freeze.json').read_text())
    for name, value in freeze['reference_files'].items():
        assert digest(REPO / name) == value, name
    for name, spec in freeze['runtime_dependencies'].items():
        assert digest(REPO / name) == spec['sha256'], name
    protocol = json.loads((HERE / 'protocol_freeze.json').read_text())
    for name, value in protocol['files'].items():
        assert digest(HERE / name) == value, name
    return freeze, protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:11436')
    args = parser.parse_args()
    freeze, protocol = verify_freeze()
    cases = read_jsonl(HERE / 'cases.jsonl')
    truths = read_jsonl(HERE / 'ground_truth.jsonl')
    assert len(cases) == len(truths) == 48
    by_truth = {row['case_id']: row for row in truths}
    providers = {name: OllamaProvider('qwen2.5:3b', base_url=args.base_url, seed=17)
                 for name in ('baseline', 'defended')}
    for provider in providers.values():
        assert provider.identity['model_digest'] == freeze['model']['model_digest']
        assert provider.identity['options'] == freeze['model']['options']
    config = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'reference_commit': freeze['reference_commit'], 'protocol': protocol,
              'provider': providers['baseline'].identity,
              'same_model_two_transport_instances': True,
              'call_limit': 120, 'maximum_planned_calls': 112,
              'note': 'Host abstentions are outcomes, not fabricated model responses. No dummy calls.'}
    with (HERE / 'execution_config.json').open('x') as stream:
        json.dump(config, stream, indent=2)
        stream.write('\n')
    authorization = Authorization()
    assistants = {'baseline': BaselineAssistant(providers['baseline'], authorization),
                  'defended': DefendedAssistant(providers['defended'], authorization)}
    streams, records, repeated = {}, [], []
    started = time.perf_counter()
    try:
        for system in assistants:
            streams[system] = (HERE / (system + '_results.jsonl')).open('x')
            streams['repeat_' + system] = (HERE / ('repeat_' + system + '_results.jsonl')).open('x')
        repeat_ids = set(protocol['repeat_case_ids'])
        schedule = [('primary', case) for case in cases]
        schedule += [('repeat', case) for case in cases if case['id'] in repeat_ids]
        for index, (phase, case) in enumerate(schedule):
            order = ('baseline', 'defended') if index % 2 == 0 else ('defended', 'baseline')
            for system in order:
                assert sum(p.calls for p in providers.values()) < 120
                before = providers[system].calls
                began = time.perf_counter()
                result = assistants[system].answer(case['question'], case['document'])
                elapsed = time.perf_counter() - began
                attempts = providers[system].calls - before
                assert attempts in (0, 1)
                row = {'case_id': case['id'], 'category': case['category'], 'system': system,
                       'phase': phase, 'question': case['question'], 'document': case['document'],
                       'result': result, 'runtime_seconds': elapsed,
                       'model_calls_attempted': attempts}
                row['scores'] = score_record(case, by_truth[case['id']], result, system)
                destination = system if phase == 'primary' else 'repeat_' + system
                streams[destination].write(json.dumps(row, sort_keys=True) + '\n')
                streams[destination].flush()
                (records if phase == 'primary' else repeated).append(row)
                print(f"{phase} {case['id']} {system}: {result['status']}", flush=True)
                if providers[system].stopped:
                    raise RuntimeError('CPU_verification_failed')
        verify_freeze()
        scores = {'primary': summarize(cases, truths, records),
                  'repeats': summarize(cases, truths, repeated),
                  'model_calls_total': sum(p.calls for p in providers.values()),
                  'runtime_seconds': time.perf_counter() - started,
                  'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        with (HERE / 'scores.json').open('x') as stream:
            json.dump(scores, stream, indent=2, sort_keys=True)
            stream.write('\n')
    finally:
        for stream in streams.values():
            stream.close()
        authorization.close()


if __name__ == '__main__':
    main()
