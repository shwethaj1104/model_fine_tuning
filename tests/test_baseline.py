"""Offline checks: input isolation, token budgets, and safe upload packaging."""
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from evaluation.run_model import load_tickets, messages_for, token_lengths
from scripts.prepare_colab import prepare


class RecordingTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        # Deliberately longer than the proposed budget. The code must report,
        # not truncate/drop, this example.
        return list(range(700 if messages[-1]['role'] == 'assistant' else 550))


class BaselineChecks(unittest.TestCase):
    def test_gold_is_absent_from_inference_messages(self):
        row = {'id': 'x', 'ticket': 'Where do I set my callback URL?', 'expected': {'secret_marker': 'GOLD_ONLY'}}
        messages = messages_for(row['ticket'], 'Return JSON')
        self.assertNotIn('GOLD_ONLY', json.dumps(messages))
        self.assertEqual([m['role'] for m in messages], ['system', 'user'])
        tokenizer = RecordingTokenizer()
        report = token_lengths(tokenizer, [row], 'Return JSON', 512)
        self.assertEqual(report['over_budget_ids'], ['x'])
        self.assertEqual(report['max_full_training_tokens'], 700)
        self.assertEqual(report['rows'][0]['input_tokens'], 550)
        self.assertNotIn('GOLD_ONLY', json.dumps(tokenizer.calls[0]))
        self.assertTrue(all(c[1]['enable_thinking'] is False for c in tokenizer.calls))

    def test_header_is_accepted_and_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rows.jsonl'
            record = json.dumps({'id': 'x', 'ticket': 'An example'}) + '\n'
            path.write_text('# Explanation\n' + record)
            self.assertEqual(len(load_tickets(path)), 1)
            path.write_text('# Explanation\n' + record * 2)
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                load_tickets(path)

    def test_upload_package_contains_no_secrets_or_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = prepare(Path(tmp) / 'project.zip')
            with ZipFile(package) as archive:
                names = archive.namelist()
                self.assertIn('ticket-triage/evaluation/run_model.py', names)
                self.assertIn('ticket-triage/data/validation.jsonl', names)
                for name in names:
                    self.assertNotIn('.env', name)
                    self.assertNotIn('__pycache__', name)
                    self.assertNotIn('/model/', name)
                    self.assertNotIn('/artifacts/', name)


if __name__ == '__main__':
    unittest.main()
