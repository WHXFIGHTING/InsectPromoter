#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

import torch

from src.io import read_records, write_tsv
from src.scanning import (
    CANDIDATE_FIELDS,
    WINDOW_133_FIELDS,
    WINDOW_300_FIELDS,
    run_scan,
)


LOGGER = logging.getLogger("insect_promoter.scan")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan long DNA sequences with the final 300 bp classifier, 133 bp classifier, and activity head."
    )
    parser.add_argument("--input", required=True, help="Input FASTA, CSV, or TSV file.")
    parser.add_argument("--output-dir", required=True, help="Directory for the three result tables.")
    parser.add_argument("--sequence-column", default="sequence", help="Sequence column for CSV/TSV input.")
    parser.add_argument("--id-column", default="id", help="Identifier column for CSV/TSV input.")
    parser.add_argument("--step-300", type=int, default=50, help="Step size for 300 bp windows.")
    parser.add_argument("--step-133", type=int, default=1, help="Step size for 133 bp windows.")
    parser.add_argument("--threshold-300", type=float, default=0.5)
    parser.add_argument("--threshold-133", type=float, default=0.5)
    parser.add_argument(
        "--top-300",
        type=int,
        default=1,
        help="Maximum passing 300 bp windows retained per input; 0 keeps all passing windows.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.step_300 <= 0 or args.step_133 <= 0:
        parser.error("window steps must be positive")
    if not 0 <= args.threshold_300 <= 1 or not 0 <= args.threshold_133 <= 1:
        parser.error("classification thresholds must be between 0 and 1")
    if args.top_300 < 0:
        parser.error("--top-300 must be zero or a positive integer")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    records = read_records(Path(args.input), args.sequence_column, args.id_column)
    if not records:
        raise ValueError("input contains no sequences")

    device = torch.device(args.device)
    LOGGER.info("Loaded %d sequences; device=%s", len(records), device)
    windows_300, windows_133, candidates = run_scan(
        records=records,
        step_300=args.step_300,
        step_133=args.step_133,
        threshold_300=args.threshold_300,
        threshold_133=args.threshold_133,
        top_300=args.top_300,
        device=device,
        batch_size=args.batch_size,
    )

    output_dir = Path(args.output_dir)
    write_tsv(output_dir / "300bp_windows.tsv", windows_300, WINDOW_300_FIELDS)
    write_tsv(output_dir / "133bp_windows.tsv", windows_133, WINDOW_133_FIELDS)
    write_tsv(output_dir / "candidates.tsv", candidates, CANDIDATE_FIELDS)
    LOGGER.info(
        "Wrote %d 300 bp windows, %d 133 bp windows, and %d final candidates to %s",
        len(windows_300), len(windows_133), len(candidates), output_dir,
    )


if __name__ == "__main__":
    main()
