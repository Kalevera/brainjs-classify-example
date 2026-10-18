import json
import tempfile
import unittest
from pathlib import Path
from classifier.data import (DataError, Record, canonicalize_records, encode_labels,
                             encode_text, fit_vocabulary, label_vocabulary,
                             load_records, split_records)


class DataTests(unittest.TestCase):
    def load(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.json'
            path.write_text(json.dumps(rows))
            return load_records(path)

    def test_legacy_formats_and_phrase_labels(self):
        records = self.load([
            {'input': 'What is my name?', 'output': 'Question my name'},
            {'input': {'Who is Jim Carey?': 1}, 'output': {'Question': 1, 'Jim Carey': 1, 'absent': 0}},
            {'input': ['Hello'], 'output': ['Declaration', 'Greeting']},
        ])
        self.assertEqual(records[0].labels, ('my', 'name', 'question'))
        self.assertEqual(records[1].labels, ('jim carey', 'question'))
        self.assertEqual(records[2].text, 'Hello')

    def test_reject_ambiguous_or_malformed_rows(self):
        for row in [
            {'input': ['one', 'two'], 'output': 'question'},
            {'input': {'one': 1, 'two': 1}, 'output': 'question'},
            {'input': 'hello', 'output': {'': 1}},
            {'input': 'hello', 'output': {'question': float('nan')}},
            {'input': 'hello', 'output': {'question': 0.5}},
            {'input': 'hello', 'output': []},
            {'input': '', 'output': 'question'},
        ]:
            with self.subTest(row=row), self.assertRaises(DataError):
                self.load([row])

    def test_duplicate_conflicts_are_reported_not_unioned(self):
        rows = [Record(' HELLO ', ('a',)), Record('hello', ('a',)),
                Record('Why?', ('question',)), Record('why?', ('declaration',))]
        clean, conflicts = canonicalize_records(rows)
        self.assertEqual(clean, [Record('hello', ('a',))])
        self.assertEqual(len(conflicts), 1)
        with self.assertRaises(DataError):
            split_records(rows)

    def test_split_has_no_normalized_duplicate_leakage(self):
        rows = [Record('A', ('a',)), Record(' a ', ('a',)), Record('b', ('b',)), Record('c', ('c',))]
        train, validation = split_records(rows, 0.5)
        self.assertEqual(len(train) + len(validation), 3)
        self.assertFalse({r.text for r in train} & {r.text for r in validation})
        self.assertEqual((train, validation), split_records(rows, 0.5))

    def test_long_input_keeps_question_features_at_the_end(self):
        vocabulary = ['word:question', 'word:?']
        text = ('unseenprefix ' * 1000) + 'question?'
        self.assertEqual(encode_text(text, vocabulary), [1.0, 1.0])

    def test_features_are_binary_and_train_only(self):
        train = [Record('Hello cat', ('question', 'my'))]
        vocabulary = fit_vocabulary(train)
        self.assertNotIn('word:unseen', vocabulary)
        self.assertIn('word:cat', vocabulary)
        self.assertIn('char3:cat', vocabulary)
        self.assertEqual(encode_text('HELLO cat cat', vocabulary), encode_text('hello cat', vocabulary))
        self.assertTrue(set(encode_text('unseen', vocabulary)) <= {0.0, 1.0})
        labels = label_vocabulary(train)
        self.assertEqual(labels, ['my', 'question'])
        self.assertEqual(encode_labels(('question',), labels), [0.0, 1.0])
        with self.assertRaises(DataError):
            encode_labels(('unknown',), labels)


if __name__ == '__main__':
    unittest.main()
