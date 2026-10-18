"""Validated legacy datasets and deterministic, training-only text features."""
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass


class DataError(ValueError):
    pass


@dataclass(frozen=True)
class Record:
    text: str
    labels: tuple


def normalize_text(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def _binary(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value in (0, 1)


def _text(value):
    if isinstance(value, dict):
        if len(value) != 1 or not all(_binary(v) and v == 1 for v in value.values()):
            raise DataError('input mapping must contain one sentence with value 1')
        value = next(iter(value))
    elif isinstance(value, list):
        if len(value) != 1:
            raise DataError('input array must contain exactly one sentence')
        value = value[0]
    if not isinstance(value, str) or not normalize_text(value):
        raise DataError('input must be a nonempty sentence')
    return value.strip()


def _labels(value):
    if isinstance(value, str):
        value = value.split()
    elif isinstance(value, dict):
        if not all(_binary(v) for v in value.values()):
            raise DataError('output mapping values must be finite binary numbers')
        if any(not isinstance(k, str) or not k.strip() for k in value):
            raise DataError('output mapping contains an empty or non-string label')
        value = [k for k, v in value.items() if v == 1]
    if not isinstance(value, list) or not value:
        raise DataError('output must contain at least one label')
    if any(not isinstance(label, str) or not label.strip() for label in value):
        raise DataError('labels must be nonempty strings')
    return tuple(sorted({normalize_text(label) for label in value}))


def load_records(path):
    try:
        with open(path, encoding='utf-8') as source:
            rows = json.load(source)
    except (OSError, ValueError) as exc:
        raise DataError('cannot read dataset: {}'.format(exc)) from exc
    if not isinstance(rows, list) or not rows:
        raise DataError('dataset must be a nonempty array')
    result = []
    for index, row in enumerate(rows):
        try:
            if not isinstance(row, dict) or set(row) != {'input', 'output'}:
                raise DataError('row must have exactly input and output fields')
            result.append(Record(_text(row['input']), _labels(row['output'])))
        except DataError as exc:
            raise DataError('{} row {}: {}'.format(path, index + 1, exc)) from exc
    return result


def canonicalize_records(records):
    """Remove repeated examples; exclude and report contradictory annotations."""
    groups = defaultdict(list)
    for record in records:
        groups[normalize_text(record.text)].append(record)
    clean, conflicts = [], []
    for text, group in sorted(groups.items()):
        annotations = sorted({record.labels for record in group})
        if len(annotations) > 1:
            conflicts.append({'text': text, 'annotations': [list(a) for a in annotations]})
        else:
            clean.append(Record(text, annotations[0]))
    return clean, conflicts


def split_records(records, validation_fraction=0.2, seed=42):
    if not 0 <= validation_fraction < 1:
        raise DataError('validation_fraction must be between 0 inclusive and 1 exclusive')
    clean, conflicts = canonicalize_records(records)
    if conflicts:
        raise DataError('conflicting annotations must be resolved before splitting')
    random.Random(seed).shuffle(clean)
    count = int(len(clean) * validation_fraction)
    if validation_fraction and len(clean) > 1:
        count = max(1, min(count, len(clean) - 1))
    return clean[count:], clean[:count]


def features(text):
    normalized = normalize_text(text)
    words = re.findall(r"\w+(?:'\w+)?|[^\w\s]", normalized)
    padded = '^' + normalized + '$'
    return {'word:' + word for word in words} | {'char3:' + padded[i:i + 3] for i in range(max(0, len(padded) - 2))}


def fit_vocabulary(train, max_features=2048, min_count=1):
    if max_features < 1 or min_count < 1:
        raise DataError('feature limits must be positive')
    counts = Counter(feature for row in train for feature in features(row.text))
    ranked = sorted((feature for feature, count in counts.items() if count >= min_count), key=lambda feature: (-counts[feature], feature))
    return sorted(ranked[:max_features])


def encode_text(text, vocabulary):
    active = features(text)
    return [float(feature in active) for feature in vocabulary]


def label_vocabulary(records):
    return sorted({label for record in records for label in record.labels})


def encode_labels(labels, label_names):
    if set(labels) - set(label_names):
        raise DataError('labels absent from saved vocabulary: {}'.format(sorted(set(labels) - set(label_names))))
    return [float(label in labels) for label in label_names]
