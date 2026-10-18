import unittest
from npb_integrated.src.extraction import extract


class ExtractionTests(unittest.TestCase):
    def test_numeric_facts_and_exact_spans(self):
        for question, source, expected in [
            ('What was revenue growth?', 'Revenue increased by 12%.', '12%'),
            ('What was revenue growth?', 'Revenue decreased by 8.50%.', '-8.5%'),
            ('What was revenue?', 'Revenue was $1,200.50 million.', '$1200.5 million'),
            ('What was net income?', 'Net income was $8 million.', '$8 million'),
        ]:
            with self.subTest(source=source):
                result = extract(question, '\n  ' + source + '\n')
                fact = result['facts'][0]
                self.assertEqual(result['status'], 'answered')
                self.assertEqual(fact['value'], expected)
                self.assertEqual(('\n  ' + source + '\n')[fact['start']:fact['end']], source)

    def test_counterfactual_fact_change_changes_value(self):
        a = extract('What was revenue growth?', 'Revenue increased by 12%.')
        b = extract('What was revenue growth?', 'Revenue increased by 15%.')
        self.assertNotEqual(a['facts'][0]['value'], b['facts'][0]['value'])

    def test_conflicts_are_not_silently_selected(self):
        result = extract('What was revenue?', 'Revenue was $120 million. Revenue was $150 million.')
        self.assertEqual(result['status'], 'ambiguous')
        self.assertEqual(len(result['facts']), 2)

    def test_year_binding(self):
        doc = 'In 2024, revenue was $100 million. In 2025, revenue was $120 million.'
        self.assertEqual(extract('What was revenue in 2025?', doc)['facts'][0]['value'], '$120 million')
        self.assertEqual(extract('What was revenue?', doc)['status'], 'ambiguous')

    def test_predictions_and_opinions_are_not_reported_actuals(self):
        for doc in ['Revenue could reach $900 million.', 'The analyst expects revenue of $900 million.',
                    'The author recommends buying at $20.']:
            self.assertEqual(extract('What was revenue?', doc)['status'], 'unsupported')

    def test_unparsed_possible_conflict_is_surfaced(self):
        self.assertEqual(extract('What was revenue?',
            'Revenue was $120 million. Revenue totals $190 million.')['status'], 'ambiguous')

    def test_quoted_numeric_statement_conservatively_reports_ambiguity(self):
        result = extract('What was revenue growth?',
                         'The guide says "Revenue increased by 50%."\nRevenue increased by 12%.')
        self.assertEqual(result['facts'][0]['value'], '12%')
        self.assertEqual(len(result['facts']), 1)
        self.assertEqual(result['status'], 'ambiguous')

    def test_invalid_inputs(self):
        for question, document in [(None, 'a'), ('x', ''), ('x', 'a' * 32001), ('x', '\ud800')]:
            with self.assertRaises((ValueError, UnicodeError)):
                extract(question, document)

    def test_unsupported_question_is_explicit(self):
        self.assertEqual(extract('Should I buy this stock?', 'Revenue was $120 million.')['status'], 'unsupported')
