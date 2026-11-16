#!/usr/bin/env python3
"""Screen long sequences using the three-model peak-consistency workflow."""

import argparse
from itertools import islice
import logging
import math
from pathlib import Path
import time

import torch

from src.config import MAX_LENGTH, MODEL_SPECS
from src.inference import add_activity_scores, add_classification_scores
from src.io import FASTA_SUFFIXES, read_fasta, write_tsv
from src.model import ModelRunner
from src.preprocessing import prepare_sequences, VALID_BASES
from src.scanning import validate_scan_inputs, window_starts


LOGGER = logging.getLogger("insect_promoter.screen")
FIELDS = [
    "input_id", "input_length", "strand", "passed", "passed_model_filters", "rejection_reasons",
    "parent_300bp_start", "parent_300bp_end", "promoter_score_300bp", "model_tss_300bp",
    "class_peak_start", "class_peak_end", "max_promoter_score_133bp", "model_tss_class_peak",
    "start", "end", "promoter_score_133bp", "model_tss_activity_peak",
    "activity_standardized", "activity_ln", "activity_linear",
    "class_start_in_parent0", "activity_start_in_parent0",
    "class_expected_delta_bp", "activity_expected_delta_bp", "peak_delta_bp",
    "gc_fraction", "sequence",
]
CANDIDATE_FIELDS = [
    "id", "sequence", "length", "class_score", "activity_linear",
    "strand", "start", "end", "class_shift", "activity_shift", "peak_gap",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="FASTA with upstream sequences (at least 300 bp each).")
    parser.add_argument("--output-dir", default="screen_results")
    parser.add_argument("--step-300", type=int, default=10)
    parser.add_argument("--step-133", type=int, default=10)
    parser.add_argument("--threshold-300", type=float, default=0.8, help="Minimum 300 bp classification probability (inclusive).")
    parser.add_argument("--threshold-133", type=float, default=0.9, help="Minimum probability at BOTH the classification and activity peaks (inclusive).")
    parser.add_argument("--min-activity", type=float, default=5.0, help="Strict lower bound on predicted LINEAR basal activity; equality fails.")
    parser.add_argument("--core-offset", type=int, default=134, help="Expected 0-based core start inside the 300 bp parent: 200 - 66 = 134.")
    parser.add_argument("--max-core-shift", type=int, default=20, help="Maximum distance of EACH core peak from --core-offset.")
    parser.add_argument("--max-peak-shift", type=int, default=10, help="Maximum start-position distance between classification and activity peaks.")
    parser.add_argument("--both-strands", action="store_true", help="Also scan the reverse complement; default scans the input direction.")
    parser.add_argument("--exclude-fasta", help="Optional previously synthesized core sequences to exclude, including reverse complements.")
    parser.add_argument("--deduplicate", action="store_true", help="Keep the first passing occurrence of identical/reverse-complement core sequences.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    for option, value in (("--input", args.input), ("--exclude-fasta", args.exclude_fasta)):
        if value and Path(value).suffix.lower() not in FASTA_SUFFIXES:
            parser.error(f"{option} requires a FASTA file (.fa, .fasta, .fna, or .fas)")
    if min(args.step_300, args.step_133, args.batch_size) <= 0:
        parser.error("steps and batch size must be positive")
    if not all(0 <= value <= 1 for value in (args.threshold_300, args.threshold_133)):
        parser.error("classification thresholds must be between 0 and 1")
    if not math.isfinite(args.min_activity) or args.min_activity < 0:
        parser.error("--min-activity must be finite and nonnegative")
    if not 0 <= args.core_offset <= 167 or min(args.max_core_shift, args.max_peak_shift) < 0:
        parser.error("core offset must be 0..167; peak-distance limits must be nonnegative")
    return args


def reverse_complement(sequence):
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def canonical(sequence):
    return min(sequence, reverse_complement(sequence))


def input_coordinates(start0, length, input_length, strand):
    if strand == "+":
        return start0 + 1, start0 + length
    return input_length - start0 - length + 1, input_length - start0


def result_base(record, strand):
    row = dict.fromkeys(FIELDS, "")
    row.update(input_id=record["id"], input_length=len(record["sequence"]), strand=strand,
               passed=0, passed_model_filters=0)
    return row


def select_core_peaks(parent, cores, args):
    # Select both maxima before applying ANY core-window filters; ties keep the first start.
    classification = max(cores, key=lambda row: row["promoter_score_133bp"])
    activity = max(cores, key=lambda row: row["activity_linear"])
    cs, acs = classification["start_in_parent0"], activity["start_in_parent0"]
    cd, ad, pd = abs(cs - args.core_offset), abs(acs - args.core_offset), abs(cs - acs)
    checks = {
        "300bp_probability": parent["promoter_score_300bp"] >= args.threshold_300,
        "classification_peak_probability": classification["promoter_score_133bp"] >= args.threshold_133,
        "activity_window_probability": activity["promoter_score_133bp"] >= args.threshold_133,
        "linear_activity": activity["activity_linear"] > args.min_activity,
        "classification_peak_position": cd <= args.max_core_shift,
        "activity_peak_position": ad <= args.max_core_shift,
        "peak_distance": pd <= args.max_peak_shift,
    }
    row = dict(parent)
    row.update(
        class_peak_start=classification["start"], class_peak_end=classification["end"],
        max_promoter_score_133bp=classification["promoter_score_133bp"],
        model_tss_class_peak=input_coordinates(classification["start_in_oriented_input0"] + 66, 1,
                                               parent["input_length"], parent["strand"])[0],
        model_tss_activity_peak=input_coordinates(activity["start_in_oriented_input0"] + 66, 1,
                                                  parent["input_length"], parent["strand"])[0],
        class_start_in_parent0=cs, activity_start_in_parent0=acs,
        class_expected_delta_bp=cd, activity_expected_delta_bp=ad, peak_delta_bp=pd,
        passed_model_filters=int(all(checks.values())),
    )
    for field in ("start", "end", "promoter_score_133bp", "activity_standardized", "activity_ln", "activity_linear", "sequence"):
        row[field] = activity[field]
    sequence = activity["sequence"]
    row["gc_fraction"] = (sequence.count("G") + sequence.count("C")) / 133
    row["passed"] = int(all(checks.values()))
    row["rejection_reasons"] = ";".join(name for name, passed in checks.items() if not passed)
    return row


def window_record(record, strand, start0, length, offset=""):
    start, end = input_coordinates(start0, length, len(record["sequence"]), strand)
    return dict(input_id=record["id"], strand=strand, start=start, end=end,
                start_in_oriented_input0=start0, start_in_parent0=offset)


def score_windows(windows, model_name, args, device, stats):
    iterator = iter(windows)
    batch = list(islice(iterator, args.batch_size))
    if not batch:
        return
    LOGGER.info("Scoring %s", model_name)
    runner = ModelRunner(MODEL_SPECS[model_name], device)
    started, count, token_maximum = time.monotonic(), 0, 0
    try:
        while batch:
            lengths = [len(ids) for ids in runner.tokenizer(
                prepare_sequences(batch, runner.spec["preprocess"]), truncation=False)["input_ids"]]
            token_maximum = max(token_maximum, max(lengths))
            if token_maximum > MAX_LENGTH:
                raise ValueError(f"{model_name}: observed {token_maximum} tokens exceeds {MAX_LENGTH}; refusing to truncate")
            logits = runner.predict_logits(batch, args.batch_size).float()
            if not torch.isfinite(logits).all():
                raise ValueError(f"{model_name}: nonfinite model output")
            if runner.spec["task"] == "classification":
                add_classification_scores(batch, logits, runner.spec["length"])
            else:
                add_activity_scores(batch, logits)
            count += len(batch)
            if count % (args.batch_size * 128) == 0:
                LOGGER.info("%s: %d windows", model_name, count)
            yield from batch
            batch = list(islice(iterator, args.batch_size))
    finally:
        stats[model_name] = dict(n_windows=count, maximum_tokens=token_maximum,
                                 seconds=time.monotonic() - started)
        del runner
        if device.type == "cuda":
            torch.cuda.empty_cache()


def run_screen(records, args, device):
    parents, stats = {}, {}

    def windows300():
        for index, record in enumerate(records):
            for strand in (["+", "-"] if args.both_strands else ["+"]):
                sequence = record["sequence"] if strand == "+" else reverse_complement(record["sequence"])
                for start0 in window_starts(len(sequence), 300, args.step_300):
                    window = window_record(record, strand, start0, 300)
                    window.update(_input_index=index, sequence=sequence[start0:start0 + 300])
                    yield window

    for window in score_windows(windows300(), "classifier-300bp", args, device, stats):
        key = window["_input_index"], window["strand"]
        if key not in parents or window["promoter_score_300bp"] > parents[key]["promoter_score_300bp"]:
            parents[key] = window

    children = {}
    for key, parent in parents.items():
        if parent["promoter_score_300bp"] < args.threshold_300:
            continue
        children[key] = []
        record = records[key[0]]
        for offset in window_starts(300, 133, args.step_133):
            window = window_record(record, parent["strand"], parent["start_in_oriented_input0"] + offset, 133, offset)
            window.update(_key=key, sequence=parent["sequence"][offset:offset + 133])
            children[key].append(window)
    cores = [window for group in children.values() for window in group]
    list(score_windows(cores, "classifier-133bp", args, device, stats))
    list(score_windows(cores, "activity-133bp", args, device, stats))

    excluded = set()
    if args.exclude_fasta:
        excluded_records = read_fasta(Path(args.exclude_fasta))
        for record in excluded_records:
            if len(record["sequence"]) != 133 or set(record["sequence"]) - VALID_BASES:
                raise ValueError("--exclude-fasta must contain valid 133 bp core sequences")
            excluded.add(canonical(record["sequence"]))
    seen, results = set(), []
    for key, parent in parents.items():
        record = records[key[0]]
        row = result_base(record, parent["strand"])
        row.update(parent_300bp_start=parent["start"], parent_300bp_end=parent["end"],
                   promoter_score_300bp=parent["promoter_score_300bp"],
                   model_tss_300bp=input_coordinates(parent["start_in_oriented_input0"] + 200, 1,
                                                     len(record["sequence"]), parent["strand"])[0])
        if key not in children:
            row["rejection_reasons"] = "300bp_probability"
        else:
            row = select_core_peaks(row, children[key], args)
            sequence_key = canonical(row["sequence"])
            reasons = [row["rejection_reasons"]] if row["rejection_reasons"] else []
            if sequence_key in excluded:
                reasons.append("previously_synthesized_sequence")
            if args.deduplicate and row["passed"] and not reasons and sequence_key in seen:
                reasons.append("duplicate_sequence")
            if reasons:
                row["passed"] = 0
                row["rejection_reasons"] = ";".join(reasons)
            if row["passed"]:
                seen.add(sequence_key)
        results.append(row)
    return results, stats


def write_candidates(path, candidates):
    with path.open("w") as handle:
        for row in candidates:
            metadata = [
                f"length={len(row['sequence'])}",
                f"class_score={row['promoter_score_133bp']:.6f}",
                f"activity_linear={row['activity_linear']:.6f}",
                f"strand={row['strand']}",
                f"position={row['start']}-{row['end']}",
                f"class_shift={row['class_expected_delta_bp']}",
                f"activity_shift={row['activity_expected_delta_bp']}",
                f"peak_gap={row['peak_delta_bp']}",
            ]
            handle.write(f">{row['input_id']}|{'|'.join(metadata)}\n{row['sequence']}\n")


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    records = read_fasta(Path(args.input))
    if not records:
        raise ValueError("input contains no sequences")
    validate_scan_inputs(records)
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("input identifiers must be unique")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    started = time.monotonic()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Screening %d regions on %s in FP32", len(records), device)
    results, stats = run_screen(records, args, device)
    candidates = [row for row in results if row["passed"]]
    output_path = output_dir / "candidates.fa"
    write_candidates(output_path, candidates)
    table_rows = [{
        "id": row["input_id"], "sequence": row["sequence"], "length": len(row["sequence"]),
        "class_score": round(row["promoter_score_133bp"], 6),
        "activity_linear": round(row["activity_linear"], 6),
        "strand": row["strand"], "start": row["start"], "end": row["end"],
        "class_shift": row["class_expected_delta_bp"],
        "activity_shift": row["activity_expected_delta_bp"], "peak_gap": row["peak_delta_bp"],
    } for row in candidates]
    write_tsv(output_dir / "candidates.tsv", table_rows, CANDIDATE_FIELDS)
    for model_name, stage in stats.items():
        LOGGER.info("%s: %d windows, maximum %d tokens", model_name, stage["n_windows"], stage["maximum_tokens"])
    LOGGER.info("Passed: %d/%d regions, %d/%d orientations; %.1f seconds; outputs: %s and %s",
                len({row["input_id"] for row in candidates}), len(records), len(candidates),
                len(results), time.monotonic() - started, output_path, output_dir / "candidates.tsv")


if __name__ == "__main__":
    main()
