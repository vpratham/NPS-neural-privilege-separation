"""Reproduce a saved host-preflight failure without any model generation."""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent.parent)]
from npb_integrated.src.defended import DefendedAssistant
from npb_integrated.src.extraction import extract


class NoInference:
    def generate(self, _messages):
        raise AssertionError('This reproduction must stop at host preflight.')


def reproduce(case, recorded):
    assert recorded['result']['generation'] is None, 'This case requires model inference.'
    assistant = DefendedAssistant(provider=NoInference())
    try:
        result = assistant.answer(case['question'], case['document'])
        for field in ('status', 'answer', 'reason', 'extraction'):
            assert result[field] == recorded['result'][field], field
        clean = case['document']
        for span in sorted(case['attacker_spans'], key=lambda s: s['start'], reverse=True):
            assert clean[span['start']:span['end']] == span['text']
            clean = clean[:span['start']] + clean[span['end']:]
        return {'kind': 'frozen_host_preflight_no_model', 'case_id': case['id'],
                'reproduced': True, 'result': result,
                'attacker_span_removal': {'document': clean, 'preflight': extract(case['question'], clean)}
                if case['attacker_spans'] else None}
    finally:
        assistant.authorization.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True)
    args = parser.parse_args()
    cases = {r['id']: r for r in map(json.loads, (HERE / 'cases.jsonl').read_text().splitlines())}
    rows = {r['case_id']: r for r in map(json.loads, (HERE / 'defended_results.jsonl').read_text().splitlines())}
    print(json.dumps(reproduce(cases[args.case], rows[args.case]), indent=2))


if __name__ == '__main__':
    main()
