"""Check that LoRA targets cannot silently include inputs or lose answers."""
import unittest
from training.train_lora import encode_example


class Tokenizer:
    unk_token_id = -1
    def apply_chat_template(self, messages, **kwargs):
        assert messages[-1]['role'] == 'user'
        assert kwargs['enable_thinking'] is False
        assert kwargs['add_generation_prompt'] is True
        return [10, 11, 12]
    def encode(self, text, **kwargs):
        assert kwargs['add_special_tokens'] is False
        return [20, 21]
    def convert_tokens_to_ids(self, text):
        return 99 if text == '<|im_end|>' else -1


class TrainingLabelChecks(unittest.TestCase):
    def setUp(self):
        self.row = {'id':'train-x','ticket':'I receive a 401.', 'expected':{'category':'authentication','environment':'unknown','customer_reported_impact':'The request receives a 401.','suggested_team':'developer_support'}}
    def test_only_answer_and_end_of_turn_are_supervised(self):
        encoded = encode_example(Tokenizer(), self.row, 'Instructions', 512)
        self.assertEqual(encoded['input_ids'], [10,11,12,20,21,99])
        self.assertEqual(encoded['labels'], [-100,-100,-100,20,21,99])
        self.assertEqual(encoded['attention_mask'], [1]*6)
        self.assertEqual(len(encoded['labels']), len(encoded['input_ids']))
    def test_overlength_is_rejected_instead_of_truncated(self):
        with self.assertRaisesRegex(ValueError, 'no truncation'):
            encode_example(Tokenizer(), self.row, 'Instructions', 5)
    def test_missing_answer_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Invalid expected'):
            encode_example(Tokenizer(), {'id':'x','ticket':'hello'}, 'Instructions', 512)


if __name__=='__main__': unittest.main()
