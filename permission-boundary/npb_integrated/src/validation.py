"""Recompute the candidate set; model bytes cannot supply a rendered answer."""
from nps_gateway.contracts import strict_json
from .extraction import extract


def validate_proposal(question, document, raw):
    extraction = extract(question, document)
    if extraction['status'] != 'answered':
        return {'status': extraction['status'], 'reason': extraction['reason'], 'extraction': extraction}
    proposal = strict_json(raw, limit=8192)
    if type(proposal) is not dict or type(proposal.get('kind')) is not str:
        raise ValueError('invalid_proposal')
    if proposal == {'kind': 'abstain'}:
        return {'status': 'refused', 'reason': 'model_abstained', 'extraction': extraction}
    if set(proposal) != {'kind', 'fact_id'} or proposal['kind'] != 'extract' or type(proposal['fact_id']) is not str:
        raise ValueError('invalid_proposal')
    fact = next((f for f in extraction['facts'] if f['fact_id'] == proposal['fact_id']), None)
    if fact is None:
        raise ValueError('unknown_fact_id')
    if document[fact['start']:fact['end']] != fact['text']:
        raise ValueError('source_span_mismatch')
    return {'status': 'answered', 'reason': 'host_validated_source_value',
            'answer': fact['value'], 'supporting_spans': [fact], 'extraction': extraction}
