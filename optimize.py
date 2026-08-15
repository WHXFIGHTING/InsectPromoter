#!/usr/bin/env python3
import argparse
import json
import logging
from pathlib import Path

import torch

from src.config import ROOT
from src.io import read_records, write_tsv
from src.optimization import (
    PARENT_FIELDS,
    VARIANT_FIELDS,
    load_rules,
    optimize_records,
)


LOGGER = logging.getLogger("insect_promoter.optimize")
DEFAULT_RULES = ROOT / "resources" / "optimization_rules_v1.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Design higher-scoring variants of TSS-centered 133 bp core promoters."
    )
    parser.add_argument("input", help="Input FASTA, CSV, or TSV containing 133 bp sequences.")
    parser.add_argument("--output", default="optimization_results", help="Output directory.")
    parser.add_argument("--top-k", type=int, default=5, help="Variants retained per parent sequence.")
    parser.add_argument("--rules", default=str(DEFAULT_RULES), help="Optimization rules JSON file.")
    parser.add_argument("--sequence-column", default="sequence", help=argparse.SUPPRESS)
    parser.add_argument("--id-column", default="id", help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    return args


def write_fasta(path, variants):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for variant in variants:
            handle.write(
                ">{}|parent={}|rank={}|n_edits={}|promoter_score={:.6f}|activity_linear={:.6f}\n".format(
                    variant["variant_id"], variant["parent_id"], variant["rank"],
                    variant["n_edits"], variant["promoter_score"], variant["activity_linear"],
                )
            )
            handle.write(variant["sequence"] + "\n")


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    input_path = Path(args.input)
    output_dir = Path(args.output)
    rules_path = Path(args.rules)
    records = read_records(input_path, args.sequence_column, args.id_column)
    if not records:
        raise ValueError("input contains no sequences")
    rules = load_rules(rules_path)
    device = torch.device(args.device)
    LOGGER.info(
        "Loaded %d parent sequences; rules=%s; device=%s",
        len(records), rules["version"], device,
    )

    parents, selected, retained = optimize_records(records, rules, device, args.top_k)
    write_tsv(output_dir / "parent_scores.tsv", parents, PARENT_FIELDS)
    write_tsv(output_dir / "optimized_variants.tsv", selected, VARIANT_FIELDS)
    write_tsv(output_dir / "all_retained_variants.tsv", retained, VARIANT_FIELDS)
    write_fasta(output_dir / "optimized_variants.fa", selected)
    metadata = {
        "input": str(input_path.resolve()),
        "output": str(output_dir.resolve()),
        "rules": str(rules_path.resolve()),
        "rules_version": rules["version"],
        "device": str(device),
        "top_k": args.top_k,
        "n_parents": len(parents),
        "n_selected_variants": len(selected),
        "n_retained_variants": len(retained),
        "ranking": "predicted activity_linear, then promoter_score",
        "activity_note": "Predicted activity is intended for ranking and is not an absolute reporter measurement.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_info.json").write_text(json.dumps(metadata, indent=2) + "\n")
    LOGGER.info("Wrote %d optimized variants to %s", len(selected), output_dir)


if __name__ == "__main__":
    main()
