import json
from pathlib import Path
import tempfile
import unittest

import torch

from classifier.data import Record
from classifier.model import LabelNetwork, load_model, measure, save_model, text_tensor, predict_text


class ModelTests(unittest.TestCase):
    def test_json_roundtrip_preserves_scores_and_label_order(self):
        device = torch.device('cpu')
        network = LabelNetwork(2, 3, 2).eval()
        vocabulary, labels = ['word:my', 'word:question'], ['my', 'question']
        x = text_tensor(['my question'], vocabulary, device)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.json'
            save_model(path, network, vocabulary, labels, .6, {'seed': 42})
            restored, artifact = load_model(path, device)
            self.assertEqual(artifact['labels'], labels)
            self.assertEqual(artifact['threshold'], .6)
            torch.testing.assert_close(network(x), restored(x), rtol=0, atol=0)
            self.assertIsInstance(json.loads(path.read_text())['state'], dict)

    def test_all_unknown_text_abstains_despite_positive_bias(self):
        network = LabelNetwork(1, 1, 1)
        with torch.no_grad():
            for parameter in network.parameters():
                parameter.fill_(10)
        result = measure(network, [Record('unseen', ('question',))], ['word:known'], ['question'], torch.device('cpu'), .5)
        self.assertEqual(result['micro_recall'], 0)
        self.assertEqual(result['exact_match'], 0)

    def test_unseen_validation_label_counts_as_missed(self):
        network = LabelNetwork(1, 1, 1)
        with torch.no_grad():
            for parameter in network.parameters():
                parameter.zero_()
            network.layers[2].bias.fill_(10)
        result = measure(network, [Record('known', ('question', 'new'))], ['word:known'], ['question'], torch.device('cpu'), .5)
        self.assertEqual(result['micro_precision'], 1)
        self.assertEqual(result['micro_recall'], .5)
        self.assertEqual(result['exact_match'], 0)
        self.assertEqual(result['unseen_labels'], ['new'])

    def test_prediction_returns_exact_multi_hot_labels_and_literal_spans(self):
        network = LabelNetwork(1, 2, 3).eval()
        with torch.no_grad():
            for parameter in network.parameters():
                parameter.zero_()
            network.layers[2].bias.copy_(torch.tensor([10., -10., 10.]))
        artifact = {'vocabulary': ['word:my'], 'labels': ['my', 'kit', 'question'], 'threshold': .5}
        result = predict_text(network, artifact, 'Is MY name here?', torch.device('cpu'))
        self.assertEqual(result['labels'], ['my', 'question'])
        self.assertEqual(result['vector'], [1, 0, 1])
        self.assertEqual(result['matched_input_spans'], [{'label': 'my', 'text': 'MY', 'start': 3, 'end': 5}])
        unknown = predict_text(network, artifact, 'zzzzz', torch.device('cpu'))
        self.assertEqual(unknown['vector'], [0, 0, 0])
        self.assertEqual(unknown['reason'], 'no_known_features')

    def test_nonfinite_weights_rejected(self):
        network = LabelNetwork(1, 1, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.json'
            save_model(path, network, ['word:a'], ['question'], .5, {})
            artifact = json.loads(path.read_text())
            artifact['state']['layers.0.bias'][0] = float('nan')
            path.write_text(json.dumps(artifact))
            with self.assertRaisesRegex(ValueError, 'non-finite'):
                load_model(path, torch.device('cpu'))


if __name__ == '__main__':
    unittest.main()
