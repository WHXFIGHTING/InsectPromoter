# InsectPromoter

## Overview

InsectPromoter is a DNABERT-2-based framework for cross-species insect promoter discovery and design. It combines three models for 300 bp promoter-region identification, 133 bp core-promoter recognition, and basal-activity prediction, providing a sequential workflow from genome-scale scanning to candidate ranking and sequence optimization.


## Installation

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

## Sequence optimization

The input must contain TSS-centered 133 bp sequences corresponding to positions -66 to +66.

```bash
python optimize.py sequences_133bp.fa --output optimization_results
```

The selected variants are written to `optimization_results/optimized_variants.fa` and `optimization_results/optimized_variants.tsv`.

Input files may be FASTA, CSV, or TSV. DNA sequences must contain only `A`, `C`, `G`, and `T`.
