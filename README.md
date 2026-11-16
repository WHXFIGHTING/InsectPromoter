# InsectPromoter

## Overview

InsectPromoter is a DNABERT-2-based framework for cross-species insect promoter discovery and design. It combines three models for 300 bp promoter-region identification, 133 bp core-promoter recognition, and basal-activity prediction, providing a sequential workflow from genome-scale scanning to candidate ranking and sequence optimization.


## Installation

For the prebuilt Docker image, use the [Docker instructions](#docker) below.
The following steps are for running the Python scripts directly.

Python 3.8 is recommended.

```bash
pip install torch==1.13.1 transformers==4.29.2 einops==0.6.1
```

Download the model weights from Zenodo:

https://zenodo.org/records/21947526

Extract the archive and place the complete `models` directory in the project root:

```text
models/
|-- classifier_300bp/
|   |-- config.json
|   `-- pytorch_model.bin
|-- classifier_133bp/
|   |-- config.json
|   `-- pytorch_model.bin
`-- activity_133bp/
    |-- config.json
    `-- pytorch_model.bin
```

## Docker

The public Docker image is available as
[`whxfighting/insectpromoter:v1.0`](https://hub.docker.com/r/whxfighting/insectpromoter/tags?name=v1.0)
for **linux/amd64**. It bundles all three model checkpoints, tokenizer files,
and the `predict`, `scan`, `screen`, and `optimize` commands. No separate model
download or Python environment is required when using Docker.

Pull the published image:

```bash
docker pull whxfighting/insectpromoter:v1.0
docker run --rm whxfighting/insectpromoter:v1.0 --help
```

Run the following examples from the directory containing your input FASTA
files. The directory is mounted at `/work`, so outputs are saved to the same
host directory. `--user` preserves the current user's output-file ownership.
GPU execution requires an NVIDIA driver and NVIDIA Container Toolkit on the
host.

Predict 300 bp promoter regions:

```bash
docker run --rm --gpus all --network none \
  --user "$(id -u):$(id -g)" -v "$PWD:/work" \
  whxfighting/insectpromoter:v1.0 predict \
  --input sequences_300bp.fa --output predictions_300bp.tsv --model classifier-300bp
```

Run both the classification and activity models on 133 bp sequences:

```bash
docker run --rm --gpus all --network none \
  --user "$(id -u):$(id -g)" -v "$PWD:/work" \
  whxfighting/insectpromoter:v1.0 predict \
  --input sequences_133bp.fa --output predictions_133bp.tsv --model both-133bp
```

Scan long sequences:

```bash
docker run --rm --gpus all --network none \
  --user "$(id -u):$(id -g)" -v "$PWD:/work" \
  whxfighting/insectpromoter:v1.0 scan \
  --input upstream_sequences.fa --output-dir scan_results
```

Apply three-model candidate screening to long FASTA sequences:

```bash
docker run --rm --gpus all --network none \
  --user "$(id -u):$(id -g)" -v "$PWD:/work" \
  whxfighting/insectpromoter:v1.0 screen \
  --input upstream_sequences.fa --output-dir screen_results
```

Optimize TSS-centered 133 bp sequences covering positions -66 to +66:

```bash
docker run --rm --gpus all --network none \
  --user "$(id -u):$(id -g)" -v "$PWD:/work" \
  whxfighting/insectpromoter:v1.0 optimize \
  sequences_133bp.fa --output optimization_results
```

For CPU execution, omit `--gpus all` and pass `--device cpu`, for example:

```bash
docker run --rm --network none \
  --user "$(id -u):$(id -g)" -v "$PWD:/work" \
  whxfighting/insectpromoter:v1.0 predict \
  --input sequences_133bp.fa --output predictions_cpu.tsv --model both-133bp --device cpu
```

Inference can run with `--network none` after the image has been pulled.
Use `docker run --rm whxfighting/insectpromoter:v1.0 COMMAND --help` to view
the options for `predict`, `scan`, `screen`, or `optimize`.

The image uses Python 3.8.20, PyTorch 1.13.1 with CUDA 11.7, and Transformers
4.29.2. The published `v1.0` image corresponds to the validated, cleaned
`insectpromoter-20261116-clean-linux-amd64.tar.gz` release. Training datasets
are excluded from the image.

The cleaned release was validated against the original Python environment on
Linux amd64: the complete GPU workflow on an RTX 2080 Ti produced nine matching
tables (604 rows; maximum numeric difference 0) and two byte-identical FASTA
outputs. CPU predictions from all three models also matched the baseline.

## Prediction

Predict 300 bp promoter regions:

```bash
python predict.py --input sequences_300bp.fa --output predictions.tsv --model classifier-300bp
```

Run both the classifier and activity model on 133 bp core promoters:

```bash
python predict.py --input sequences_133bp.fa --output predictions.tsv --model both-133bp
```

## Long-sequence scanning

```bash
python scan.py --input upstream_sequences.fa --output-dir scan_results
```

The main result is written to `scan_results/candidates.tsv`.

## Three-model candidate screening

Use the experimental candidate-screening workflow on upstream sequences in FASTA format
(`.fa`, `.fasta`, `.fna`, or `.fas`; at least 300 bp per sequence):

```bash
python screen.py --input upstream_sequences.fa --output-dir screen_results
```

Defaults: 10 bp steps for both window sizes; 300 bp probability >= 0.8;
133 bp probability >= 0.9 at both the classification and activity peaks;
predicted linear basal activity > 5. Each core peak must start within 20 bp
of the expected position (0-based offset 134 inside the 300 bp parent),
and the two peaks must be within 10 bp of each other. Only the highest-scoring
300 bp parent is used per direction; peaks are selected before position filtering.

Adjust thresholds and distances, or scan both directions:

```bash
python screen.py --input upstream_sequences.fa --output-dir screen_results \
  --step-300 10 --step-133 10 --threshold-300 0.8 --threshold-133 0.9 \
  --min-activity 5 --core-offset 134 --max-core-shift 20 --max-peak-shift 10 \
  --both-strands
```

Outputs are `candidates.tsv` and `candidates.fa`, containing passing 133 bp sequences.
The table columns are `id`, `sequence`, `length`, `class_score`, `activity_linear`,
`strand`, `start`, `end`, `class_shift`, `activity_shift`, and `peak_gap`.
`class_score` is the candidate 133 bp classification probability. FASTA headers
include the same scores and positions. The three distances are in bp:
`class_shift` and `activity_shift` are the absolute distances of the two peaks from
the expected core start; `peak_gap` is the distance between the two peaks.
Positions are 1-based inclusive within the original input; strand is relative to
that input. The default scans only the input direction.

Optional sequence filters: `--exclude-fasta previous_cores.fa` (133 bp sequences)
and `--deduplicate`. Run `python screen.py --help` for all options.
Activity thresholds refer to model predictions, not experimental fold changes.

## Sequence optimization

The input must contain TSS-centered 133 bp sequences corresponding to positions -66 to +66.

```bash
python optimize.py sequences_133bp.fa --output optimization_results
```

The selected variants are written to `optimization_results/optimized_variants.fa` and `optimization_results/optimized_variants.tsv`.

The `predict.py`, `scan.py`, and `optimize.py` scripts also accept CSV or TSV input.
DNA sequences must contain only `A`, `C`, `G`, and `T`.
