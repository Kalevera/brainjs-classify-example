# KIT: fixed-label text classification

Roberto Vega's original 2016 Brain.js experiment is preserved in `index-stream.js`,
`index.js`, `test.js`, `data/`, and the [historical write-up](docs/legacy-experiment.md).
The maintained trainer is now a small Python/PyTorch command-line program with
CUDA training, fixed label vectors, saved models, and validation metrics. No browser
or language-model service is needed. The original Brain.js scripts are archival;
`npm test` does not run the new classifier.

Instead of predicting character codes and trying to spell `question`, the network
scores a fixed list of known labels. A threshold produces a binary multi-hot
vector. Several positions can be active: `question`, `my`, and `name` may all apply.
The saved label order maps every position to its exact label. Scores are independent
sigmoid activations, not calibrated confidence estimates, and do not sum to one.

## Run on the GPU

Python 3.10+ and a CUDA-enabled PyTorch installation are required. In the development
container, the existing `torch 2.13.0+cu126` installation was tested successfully on
a GTX 1050 Ti (4 GB). No replacement driver or ML stack was installed.

If PyTorch already works, skip installation. For a separate environment:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-gpu.txt
```

The tested CUDA 12.6 wheel is pinned because newer CUDA builds may not support
older Pascal GPUs. The CUDA version shown by `nvidia-smi` describes driver support;
it need not equal PyTorch's bundled runtime version. See the
[official PyTorch builds](https://pytorch.org/get-started/previous-versions/).
A container must expose the GPU (for Docker, commonly `--gpus all`).

Run commands from the repository root:

```sh
python3 -m classifier inspect-data --data data/classification.JSON
python3 -m classifier train --model models/classification.json
python3 -m classifier predict --model models/classification.json --text 'What is your name?'
```

Training and prediction default to `--device cuda`. An unavailable or incompatible
GPU produces an error; the program never silently trains on the CPU. CPU use is
available only through an explicit `--device cpu`. JSON parsing and text feature
preparation use the CPU; network forward passes, loss, gradients and optimizer
updates run on the selected device. This is a modest single-GPU model, not a large
training workload. The CLI reports synchronized training-loop time separately
from startup and preprocessing; it does not claim a CPU/GPU speedup.

To use the richer original token-label data:

```sh
python3 -m classifier inspect-data --data data/numericsentence.JSON
python3 -m classifier train --data data/numericsentence.JSON --drop-conflicts --model models/tokens.json
python3 -m classifier predict --model models/tokens.json --text 'What is my name?'
```

`--drop-conflicts` explicitly excludes entire groups of contradictory annotations.
Without that option training stops and asks you to review them. No source dataset
is rewritten. Trained models are local, gitignored JSON files, including the input
vocabulary, output-label order, weights, threshold, data checksum and run metrics.
Prediction reuses the saved vocabularies; it never builds a new output mapping.

Prediction prints machine-readable JSON with `label_order`, `vector`, exact
`labels`, and all `scores`. No activated label means `abstained: true`; an input
without known features also abstains. This is a decision rule, not a guarantee
that unfamiliar requests will always be rejected. `--threshold` can override the
saved decision threshold for prediction; choose thresholds on validation data,
then measure final performance on separate test data.

For repeated predictions without restarting Python or reloading the GPU model:

```sh
python3 -m classifier predict --model models/tokens.json --stdin
```

In a terminal, wait for `Ready`, then type at `sentence>` and press Enter.
Results show labels and active vector positions. Ctrl-D or Ctrl-C exits cleanly.
Piped input still returns one JSON object per nonblank line without prompts.
The same process and model stay loaded.

## Data formats and limitations

- `classification.JSON`: string input and a single string label; 160 rows reduce
  to 129 unique normalized inputs. A useful starting point for sentence type.
- `numericsentence.JSON`: string input and whitespace-separated label tokens;
  `question my name` becomes three labels. Case is normalized, so `Kit` and `kit`
  share one position. Six duplicate-input groups have contradictory annotations.
- `sentence.JSON`: a one-key sentence input mapping and binary output-label map.
  Multiword keys such as `Jim Carey` remain one label. The original file has an
  empty label at row 71 and is deliberately rejected until that annotation is
  corrected. The program does not guess what the missing label should mean.
- `tester.JSON`: one-element input array and output label array are also supported.
- Empty `readyset.JSON` and `weights.JSON` are not training datasets.

Input features are bounded binary word/character features learned only from the
training split. Duplicate normalized inputs are grouped before splitting. Labels
also come only from training data; unseen validation labels count as missed
predictions and are listed in the report. Small datasets and labels that appear
only once limit generalization. These experiments are not a sentiment benchmark
or a production-ready business classifier.

Classification and extraction are separate. A fixed-label model can recognize a
known `jim carey` label, but cannot produce a previously unseen name. Prediction
includes `matched_input_spans` only for **literal matches of activated known
labels** in the original input, with Python string start/end offsets. This is a
convenience lookup, not a trained entity extractor. Handling arbitrary new names
would need a separately annotated extraction task.

## Train the final model on all usable data

Use a split while evaluating model choices. Once those choices are settled, fit
all clean examples for the final model:

```sh
python3 -m classifier train --data data/numericsentence.JSON --drop-conflicts --validation-fraction 0 --model models/kit-full.json
python3 -m classifier predict --model models/kit-full.json --stdin
```

A full-data fit reports `fit_scope: all_clean_examples` and `validation: null`.
Its training scores measure fit, not new-sentence accuracy. Conflicting groups
still need explicit exclusion or corrected annotations. This mode includes every
label in the clean dataset in its saved output mapping. New manually entered
sentences are useful spot checks; a separate, diverse labeled test set is needed
for an honest performance estimate. More varied examples per label and consistent
annotations matter more than copying the same examples many times.

## Evaluate and test

Training reports training and held-out validation exact match, micro precision,
recall and F1. Training accuracy is not evidence of generalization. Validation
is not an independent final test set, especially after repeated tuning.

```sh
python3 -m classifier evaluate --model models/classification.json --data /path/to/independent-test.json
python3 -m unittest discover -s tests -v
```

The evaluation command cannot prove that your supplied file is independent of
training data. Use a genuinely separate annotated set for final claims. Tests
cover dataset adapters, conflict handling, leakage prevention, vector mappings,
model round trips, out-of-vocabulary abstention, and explicit device selection.
The CUDA test runs a tiny real GPU operation when CUDA is available.

The legacy datasets contain historical typos, sparse labels and inconsistent
annotations. Review labels and collect new examples before relying on the model
for real automation. Keep checkpoints, private datasets, credentials and business
training recipes out of public commits; `models/` is excluded by default.


### Question/declaration follow-up examples

`data/question-followups.JSON` adds 34 annotated examples, including reported
misses, short questions, contractions and contrasting statements. It also repeats
the two working name queries as regression examples; duplicates are removed when
combining sources. Original datasets remain unchanged. To reproduce the revised
full-data model:

```sh
python3 -m classifier train --data data/numericsentence.JSON --extra-data data/question-followups.JSON --drop-conflicts --validation-fraction 0 --model models/kit-full.json
```

`--extra-data` can be repeated. All files are combined before duplicate/conflict
checks and splits; each source checksum is recorded. The current model preserves
the previous label order. Models loaded in a running interactive process do not
refresh automatically: restart prediction to load a new checkpoint.

`data/question-holdout.JSON` contains 12 separate authored sentence-type checks.
They were not included in training. Compare only question/declaration predictions
against those annotations; unannotated topic labels are outside this check's
scope. Before and after the follow-up training, sentence-type exact match was
9/12; individual misses changed. Fixing the supplied examples does not establish
an overall generalization improvement. A broader independently collected test
set is still needed. The prior model is retained locally as
`models/kit-before-question-followups.json`.

The modern feature encoder has no maximum input sentence length or character
padding: recognized features at the end of long inputs remain visible. Its fixed
feature vocabulary limits which features can contribute, and bag features lose
much word order. The archived `fluffQuestion()` instead returns only when the
input is shorter than the training maximum, leaving equal/longer inputs undefined.
