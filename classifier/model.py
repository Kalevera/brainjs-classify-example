"""A small multi-label classifier with portable, non-executable JSON artifacts."""
import json
import re
from pathlib import Path

import torch
from torch import nn

from .data import encode_text

FORMAT = 'kit-label-classifier-v1'


class LabelNetwork(nn.Module):
    def __init__(self, inputs, hidden, outputs):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(), nn.Linear(hidden, outputs))

    def forward(self, inputs):
        return self.layers(inputs)


def text_tensor(texts, vocabulary, device):
    return torch.tensor([encode_text(text, vocabulary) for text in texts], dtype=torch.float32, device=device)


def save_model(path, network, vocabulary, labels, threshold, metadata):
    artifact = {
        'format': FORMAT,
        'vocabulary': vocabulary,
        'labels': labels,
        'hidden': network.layers[0].out_features,
        'threshold': threshold,
        'state': {key: value.detach().cpu().tolist() for key, value in network.state_dict().items()},
        'metadata': metadata,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.tmp')
    temporary.write_text(json.dumps(artifact, allow_nan=False), encoding='utf-8')
    temporary.replace(target)


def load_model(path, device):
    artifact = json.loads(Path(path).read_text(encoding='utf-8'))
    if artifact.get('format') != FORMAT:
        raise ValueError('Unsupported model format; train a new label-vector model.')
    for key in ('vocabulary', 'labels'):
        values = artifact.get(key)
        if not isinstance(values, list) or not values or not all(isinstance(v, str) and v for v in values) or len(values) != len(set(values)):
            raise ValueError('Invalid model ' + key)
    hidden = artifact.get('hidden')
    threshold = artifact.get('threshold')
    if type(hidden) is not int or hidden < 1 or not isinstance(threshold, (int, float)) or not 0 < threshold < 1:
        raise ValueError('Invalid model dimensions or threshold')
    network = LabelNetwork(len(artifact['vocabulary']), hidden, len(artifact['labels'])).to(device)
    state = {key: torch.tensor(value, dtype=torch.float32, device=device) for key, value in artifact['state'].items()}
    if not all(torch.isfinite(value).all().item() for value in state.values()):
        raise ValueError('Model contains non-finite weights')
    network.load_state_dict(state, strict=True)
    network.eval()
    return network, artifact


def measure(network, records, vocabulary, labels, device, threshold):
    if not records:
        return None
    tp = fp = fn = exact = 0
    unseen = set()
    known = set(labels)
    network.eval()
    with torch.inference_mode():
        for start in range(0, len(records), 128):
            batch = records[start:start + 128]
            x = text_tensor([r.text for r in batch], vocabulary, device)
            scores = network(x).sigmoid().cpu().tolist()
            present = x.any(dim=1).cpu().tolist()
            for record, values, has_features in zip(batch, scores, present):
                active = {label for label, score in zip(labels, values) if has_features and score >= threshold}
                truth = set(record.labels)
                tp += len(active & truth)
                fp += len(active - truth)
                fn += len(truth - active)
                exact += active == truth
                unseen.update(truth - known)
    return {
        'examples': len(records), 'exact_match': exact / len(records),
        'micro_precision': tp / (tp + fp) if tp + fp else 0.0,
        'micro_recall': tp / (tp + fn) if tp + fn else 0.0,
        'micro_f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        'unseen_labels': sorted(unseen),
    }


def predict_text(network, artifact, text, device, threshold=None):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Prediction text must not be empty')
    threshold = artifact['threshold'] if threshold is None else threshold
    if not 0 < threshold < 1:
        raise ValueError('Threshold must be between 0 and 1')
    x = text_tensor([text], artifact['vocabulary'], device)
    has_features = bool(x.any().item())
    with torch.inference_mode():
        scores = network(x).sigmoid()[0].cpu().tolist()
    vector = [int(has_features and value >= threshold) for value in scores]
    active = [label for label, bit in zip(artifact['labels'], vector) if bit]
    # Literal label matches are not learned entity extraction.
    spans = []
    for label in active:
        for match in re.finditer(r'(?<!\w)' + re.escape(label) + r'(?!\w)', text, re.IGNORECASE):
            spans.append({'label': label, 'text': match.group(), 'start': match.start(), 'end': match.end()})
    return {'labels': active, 'label_order': artifact['labels'], 'vector': vector,
            'scores': dict(zip(artifact['labels'], scores)), 'threshold': threshold,
            'abstained': not active, 'reason': 'no_known_features' if not has_features else ('below_threshold' if not active else None),
            'matched_input_spans': sorted(spans, key=lambda span: (span['start'], span['label']))}
