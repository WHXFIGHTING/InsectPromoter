#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

import torch

from src.config import MODEL_NAMES, MODEL_SPECS
from src.inference import add_predictions
from src.io import read_records, write_tsv
from src.preprocessing import validate_sequences


LOGGER = logging.getLogger("insect_promoter")


def parse_args():
    parser = argparse.ArgumentParser(description="Predict insect promoter scores from fixed-length DNA sequences.")
    parser.add_argument("--input", required=True, help="Input FASTA, CSV, or TSV file.")
    parser.add_argument("--output", required=True, help="Output TSV file.")
    parser.add_argument(
        "--model",
        required=True,
        choices=(*MODEL_NAMES, "both-133bp"),
        help="Final model or paired 133 bp heads to run.",
    )
    parser.add_argument("--sequence-column", default="sequence", help="Sequence column for CSV/TSV input.")
    parser.add_argument("--id-column", default="id", help="Identifier column for CSV/TSV input.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    records = read_records(Path(args.input), args.sequence_column, args.id_column)
    if not records:
        raise ValueError("input contains no sequences")

    model_names = (
        ("classifier-133bp", "activity-133bp")
        if args.model == "both-133bp"
        else (args.model,)
    )
    validate_sequences(records, MODEL_SPECS[model_names[0]]["length"])

    device = torch.device(args.device)
    LOGGER.info("Loaded %d sequences; device=%s", len(records), device)
    for model_name in model_names:
        LOGGER.info("Running %s", model_name)
        add_predictions(records, model_name, device, args.batch_size)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output_path = Path(args.output)
    write_tsv(output_path, records)
    LOGGER.info("Wrote predictions to %s", output_path)


if __name__ == "__main__":
    main()
