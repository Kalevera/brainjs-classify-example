import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

from .data import (canonicalize_records, encode_labels, fit_vocabulary,
                   label_vocabulary, load_records, split_records)
from .device import select_device


def emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False))


def records_for(args):
    sources = [args.data, *getattr(args, 'extra_data', [])]
    raw = [record for path in sources for record in load_records(path)]
    records, conflicts = canonicalize_records(raw)
    if conflicts and not args.drop_conflicts:
        raise ValueError(f'{len(conflicts)} inputs have conflicting labels. Review with inspect-data; '
                         'use --drop-conflicts only to explicitly exclude those groups.')
    if not records:
        raise ValueError('No unambiguous examples remain')
    if conflicts:
        print(f'Excluded {len(conflicts)} conflicting input groups.', file=sys.stderr)
    return records, {
        'source_rows': len(raw), 'unique_examples': len(records),
        'conflicting_groups_excluded': len(conflicts),
        'sources': [{'file': Path(path).name, 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()} for path in sources],
    }


def synchronize(device):
    import torch
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def train(args):
    import torch
    from torch import nn
    from .model import LabelNetwork, measure, save_model, text_tensor

    if args.log_every < 0:
        raise ValueError('Log interval must be zero or positive')
    if args.epochs < 1 or args.batch_size < 1 or args.hidden < 1 or args.max_features < 1:
        raise ValueError('Epochs, batch size, hidden units and max features must be positive')
    if not 0 < args.threshold < 1 or not 0 < args.learning_rate < 1:
        raise ValueError('Threshold and learning rate must be between 0 and 1')
    if not 0 <= args.validation_fraction < 1:
        raise ValueError('Validation fraction must be at least 0 and less than 1')
    records, audit = records_for(args)
    if args.validation_fraction and len(records) < 3:
        raise ValueError('At least three unique, unambiguous inputs are needed for a training/validation split; use --validation-fraction 0 for an explicitly full-data fit')
    training, validation = split_records(records, args.validation_fraction, args.seed)
    vocabulary = fit_vocabulary(training, max_features=args.max_features)
    labels = label_vocabulary(training)
    if not vocabulary or not labels:
        raise ValueError('Training data must contain usable text features and labels')
    device = select_device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    network = LabelNetwork(len(vocabulary), args.hidden, len(labels)).to(device)
    x = text_tensor([r.text for r in training], vocabulary, device)
    y = torch.tensor([encode_labels(r.labels, labels) for r in training], dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.learning_rate)
    loss_function = nn.BCEWithLogitsLoss()
    synchronize(device)
    started = time.perf_counter()
    for epoch in range(args.epochs):
        network.train()
        indices = torch.randperm(len(training), device=device)
        for start in range(0, len(training), args.batch_size):
            batch = indices[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(network(x[batch]), y[batch])
            if not torch.isfinite(loss).item():
                raise ValueError('Training loss is not finite; lower the learning rate or inspect the data')
            loss.backward()
            optimizer.step()
        if args.log_every and ((epoch + 1) % args.log_every == 0 or epoch + 1 == args.epochs):
            print(f'epoch={epoch + 1} last_batch_loss={loss.item():.6f}', file=sys.stderr)
    synchronize(device)
    seconds = time.perf_counter() - started
    metadata = {
        'source_sha256': hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        'seed': args.seed, 'epochs': args.epochs, 'batch_size': args.batch_size,
        'learning_rate': args.learning_rate, 'validation_fraction': args.validation_fraction,
        'fit_scope': 'all_clean_examples' if not validation else 'training_split',
        'device': str(device), 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'torch_version': str(torch.__version__), 'training_seconds': seconds, 'data_audit': audit,
        'training': measure(network, training, vocabulary, labels, device, args.threshold),
        'validation': measure(network, validation, vocabulary, labels, device, args.threshold),
    }
    save_model(args.model, network, vocabulary, labels, args.threshold, metadata)
    emit({'model': str(args.model), 'features': len(vocabulary), 'labels': labels, **metadata})


def predict(args):
    from .model import load_model, predict_text

    device = select_device(args.device)
    network, artifact = load_model(args.model, device)
    if args.stdin:
        interactive = sys.stdin.isatty()
        if interactive:
            print(f'Ready on {device}: {len(artifact["labels"])} labels loaded. Type a sentence and press Enter. Ctrl-D or Ctrl-C exits.', file=sys.stderr)
        while True:
            if interactive:
                print('sentence> ', end='', file=sys.stderr, flush=True)
            line = sys.stdin.readline()
            if not line:
                if interactive:
                    print(file=sys.stderr)
                break
            text = line.rstrip('\r\n')
            if text.strip():
                result = predict_text(network, artifact, text, device, args.threshold)
                if interactive:
                    if result['labels']:
                        print('Labels: ' + ', '.join(result['labels']))
                        print('Active positions: ' + ', '.join(str(i) for i, bit in enumerate(result['vector']) if bit))
                    else:
                        print('No labels selected (' + result['reason'] + ').')
                else:
                    emit(result)
    else:
        emit(predict_text(network, artifact, args.text, device, args.threshold))


def evaluate(args):
    from .model import load_model, measure

    records, audit = records_for(args)
    device = select_device(args.device)
    network, artifact = load_model(args.model, device)
    emit({'data_audit': audit, 'evaluation': measure(network, records, artifact['vocabulary'], artifact['labels'], device, artifact['threshold']),
          'note': 'Only an independently held-out input file measures new-data performance; this command does not prove independence.'})


def main():
    parser = argparse.ArgumentParser(description='KIT fixed-label text classifier. GPU training is the default.')
    commands = parser.add_subparsers(dest='command', required=True)
    inspect = commands.add_parser('inspect-data', help='Validate and report duplicate/conflicting annotations without importing a GPU workload')
    inspect.add_argument('--data', required=True)
    for command in ('train', 'predict', 'evaluate'):
        sub = commands.add_parser(command)
        sub.add_argument('--model', default='models/kit.json')
        sub.add_argument('--device', choices=('cuda', 'cpu'), default='cuda', help='CPU requires explicit opt-in; CUDA never silently falls back')
        if command in ('train', 'evaluate'):
            sub.add_argument('--data', default='data/classification.JSON')
            sub.add_argument('--extra-data', action='append', default=[], help='Additional annotated dataset; repeat to combine files before deduplication and splitting')
            sub.add_argument('--drop-conflicts', action='store_true')
        if command == 'train':
            sub.add_argument('--epochs', type=int, default=200)
            sub.add_argument('--batch-size', type=int, default=64)
            sub.add_argument('--hidden', type=int, default=64)
            sub.add_argument('--max-features', type=int, default=2048)
            sub.add_argument('--learning-rate', type=float, default=.01)
            sub.add_argument('--validation-fraction', type=float, default=.2, help='Fraction reserved for validation; 0 fits every clean example with no held-out score')
            sub.add_argument('--seed', type=int, default=42)
            sub.add_argument('--threshold', type=float, default=.5)
            sub.add_argument('--log-every', type=int, default=50)
        if command == 'predict':
            source = sub.add_mutually_exclusive_group(required=True)
            source.add_argument('--text')
            source.add_argument('--stdin', action='store_true', help='One input per line; loads the GPU model once and emits JSON lines')
            sub.add_argument('--threshold', type=float)
    args = parser.parse_args()
    try:
        if args.command == 'inspect-data':
            raw = load_records(args.data)
            records, conflicts = canonicalize_records(raw)
            emit({'rows': len(raw), 'unique_unambiguous_inputs': len(records), 'labels': label_vocabulary(records), 'conflicts': conflicts})
        else:
            {'train': train, 'predict': predict, 'evaluate': evaluate}[args.command](args)
    except KeyboardInterrupt:
        print('\nStopped.', file=sys.stderr)
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        parser.exit(2, f'error: {error}\n')


if __name__ == '__main__':
    main()
