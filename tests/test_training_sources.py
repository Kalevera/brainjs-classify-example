import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from classifier.__main__ import records_for


class TrainingSourceTests(unittest.TestCase):
    def test_sources_merge_before_conflict_detection_and_preserve_checksums(self):
        with tempfile.TemporaryDirectory() as directory:
            first, extra = Path(directory) / 'original.json', Path(directory) / 'extra.json'
            first.write_text(json.dumps([{'input': 'Time?', 'output': 'information time'}, {'input': 'hello.', 'output': 'declaration'}]))
            extra.write_text(json.dumps([{'input': 'time?', 'output': 'question time'}, {'input': 'who?', 'output': 'question'}]))
            args = SimpleNamespace(data=first, extra_data=[extra], drop_conflicts=False)
            with self.assertRaisesRegex(ValueError, 'conflicting labels'):
                records_for(args)
            args.drop_conflicts = True
            records, audit = records_for(args)
            self.assertEqual({record.text for record in records}, {'hello.', 'who?'})
            self.assertEqual(audit['source_rows'], 4)
            self.assertEqual(audit['conflicting_groups_excluded'], 1)
            self.assertEqual([source['file'] for source in audit['sources']], ['original.json', 'extra.json'])
            self.assertTrue(all(len(source['sha256']) == 64 for source in audit['sources']))
