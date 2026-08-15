import math

import torch

from .config import ACTIVITY_LABEL_MEAN, ACTIVITY_LABEL_STD, MODEL_SPECS
from .model import ModelRunner
from .preprocessing import VALID_BASES


WINDOW_300_FIELDS = [
    "input_id", "input_length", "start", "end", "window_length",
    "promoter_score_300bp", "rank_300bp", "selected_for_133bp", "sequence",
]
WINDOW_133_FIELDS = [
    "input_id", "input_length", "start", "end", "window_length",
    "parent_300bp_start", "parent_300bp_end", "promoter_score_300bp",
    "promoter_score_133bp", "passed_133bp", "activity_standardized",
    "activity_ln", "activity_linear", "sequence",
]
CANDIDATE_FIELDS = [
    "input_id", "input_length", "start", "end", "window_length",
    "parent_300bp_start", "parent_300bp_end", "promoter_score_300bp",
    "promoter_score_133bp", "activity_standardized", "activity_ln",
    "activity_linear", "activity_rank", "sequence",
]


def window_starts(sequence_length, window_length, step):
    if sequence_length < window_length:
        return []
    starts = list(range(0, sequence_length - window_length + 1, step))
    terminal = sequence_length - window_length
    if starts[-1] != terminal:
        starts.append(terminal)
    return starts


def validate_scan_inputs(records):
    for record in records:
        sequence = record["sequence"]
        if len(sequence) < 300:
            raise ValueError(f"{record['id']}: expected at least 300 bp, observed {len(sequence)} bp")
        invalid = set(sequence) - VALID_BASES
        if invalid:
            raise ValueError(f"{record['id']}: unsupported bases: {''.join(sorted(invalid))}")


def make_300bp_windows(records, step):
    windows = []
    for input_index, record in enumerate(records):
        sequence = record["sequence"]
        for start in window_starts(len(sequence), 300, step):
            windows.append({
                "_input_index": input_index,
                "_oriented_start": start,
                "input_id": record["id"],
                "input_length": len(sequence),
                "start": start + 1,
                "end": start + 300,
                "window_length": 300,
                "sequence": sequence[start:start + 300],
            })
    return windows


def add_classification_scores(windows, model_name, device, batch_size):
    runner = ModelRunner(MODEL_SPECS[model_name], device)
    logits = runner.predict_logits(windows, batch_size)
    scores = torch.softmax(logits, dim=-1)[:, 1].tolist()
    column = "promoter_score_300bp" if model_name == "classifier-300bp" else "promoter_score_133bp"
    for window, score in zip(windows, scores):
        window[column] = score
    del runner
    if device.type == "cuda":
        torch.cuda.empty_cache()


def select_300bp_windows(windows, threshold, top_n):
    groups = {}
    for window in windows:
        groups.setdefault(window["_input_index"], []).append(window)
    selected = []
    for group in groups.values():
        ranked = sorted(group, key=lambda row: row["promoter_score_300bp"], reverse=True)
        for rank, window in enumerate(ranked, start=1):
            window["rank_300bp"] = rank
            passed = window["promoter_score_300bp"] >= threshold
            window["selected_for_133bp"] = int(passed and (top_n == 0 or rank <= top_n))
            if window["selected_for_133bp"]:
                selected.append(window)
    return selected


def make_133bp_windows(selected_300bp, step):
    deduplicated = {}
    for parent in selected_300bp:
        for offset in window_starts(300, 133, step):
            oriented_start = parent["_oriented_start"] + offset
            source_start = oriented_start + 1
            source_end = oriented_start + 133
            child = {
                "_input_index": parent["_input_index"],
                "input_id": parent["input_id"],
                "input_length": parent["input_length"],
                "start": source_start,
                "end": source_end,
                "window_length": 133,
                "parent_300bp_start": parent["start"],
                "parent_300bp_end": parent["end"],
                "promoter_score_300bp": parent["promoter_score_300bp"],
                "sequence": parent["sequence"][offset:offset + 133],
            }
            key = (child["_input_index"], source_start, source_end)
            previous = deduplicated.get(key)
            if previous is None or child["promoter_score_300bp"] > previous["promoter_score_300bp"]:
                deduplicated[key] = child
    return list(deduplicated.values())


def add_activity_scores(windows, device, batch_size):
    if not windows:
        return
    runner = ModelRunner(MODEL_SPECS["activity-133bp"], device)
    values = runner.predict_logits(windows, batch_size).reshape(-1).tolist()
    for window, standardized in zip(windows, values):
        activity_ln = standardized * ACTIVITY_LABEL_STD + ACTIVITY_LABEL_MEAN
        window["activity_standardized"] = standardized
        window["activity_ln"] = activity_ln
        window["activity_linear"] = max(0.0, math.expm1(activity_ln))
    del runner
    if device.type == "cuda":
        torch.cuda.empty_cache()


def rank_candidates(candidates):
    groups = {}
    for candidate in candidates:
        groups.setdefault(candidate["_input_index"], []).append(candidate)
    for group in groups.values():
        ranked = sorted(group, key=lambda row: row["activity_linear"], reverse=True)
        for rank, candidate in enumerate(ranked, start=1):
            candidate["activity_rank"] = rank
    return sorted(
        candidates,
        key=lambda row: (row["_input_index"], row["activity_rank"]),
    )


def run_scan(records, step_300, step_133, threshold_300, threshold_133, top_300,
             device, batch_size):
    validate_scan_inputs(records)
    windows_300 = make_300bp_windows(records, step_300)
    add_classification_scores(windows_300, "classifier-300bp", device, batch_size)
    selected_300 = select_300bp_windows(windows_300, threshold_300, top_300)

    windows_133 = make_133bp_windows(selected_300, step_133)
    if windows_133:
        add_classification_scores(windows_133, "classifier-133bp", device, batch_size)
    candidates = []
    for window in windows_133:
        window["passed_133bp"] = int(window["promoter_score_133bp"] >= threshold_133)
        window["activity_standardized"] = ""
        window["activity_ln"] = ""
        window["activity_linear"] = ""
        if window["passed_133bp"]:
            candidates.append(window)

    add_activity_scores(candidates, device, batch_size)
    return windows_300, windows_133, rank_candidates(candidates)
