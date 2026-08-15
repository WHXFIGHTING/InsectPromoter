import json
import logging
import math
import re
from pathlib import Path

import torch

from .config import ACTIVITY_LABEL_MEAN, ACTIVITY_LABEL_STD, MODEL_SPECS
from .model import ModelRunner
from .preprocessing import VALID_BASES, validate_sequences


LOGGER = logging.getLogger("insect_promoter.optimize")

PARENT_FIELDS = [
    "parent_id", "promoter_score", "activity_standardized", "activity_ln",
    "activity_linear", "gc_content", "max_homopolymer", "sequence",
]
VARIANT_FIELDS = [
    "parent_id", "variant_id", "rank", "n_edits", "n_changed_bp",
    "promoter_score", "parent_activity_linear", "activity_standardized",
    "activity_ln", "activity_linear", "delta_activity_ln",
    "delta_activity_linear", "fold_vs_parent", "gc_content",
    "max_homopolymer", "edit_description", "sequence",
]


def load_rules(path):
    path = Path(path)
    with path.open() as handle:
        rules = json.load(handle)
    required = {
        "version", "sequence_length", "coordinate_start", "slots",
        "motifs_by_zone", "search", "constraints",
    }
    missing = required - set(rules)
    if missing:
        raise ValueError("optimization rules missing fields: " + ", ".join(sorted(missing)))

    length = int(rules["sequence_length"])
    if length != MODEL_SPECS["classifier-133bp"]["length"]:
        raise ValueError("the released optimization heads require 133 bp sequences")
    coordinate_start = int(rules["coordinate_start"])
    zones = rules["motifs_by_zone"]
    seen_starts = set()
    for slot in rules["slots"]:
        start = int(slot["start"])
        zone = slot["zone"]
        index = start - coordinate_start
        if start in seen_starts:
            raise ValueError("duplicate optimization slot: {}".format(start))
        if not 0 <= index <= length - 6:
            raise ValueError("optimization slot outside sequence: {}".format(start))
        if zone not in zones:
            raise ValueError("undefined motif zone: {}".format(zone))
        seen_starts.add(start)

    for zone, motifs in zones.items():
        if not motifs:
            raise ValueError("motif zone is empty: {}".format(zone))
        for motif in motifs:
            if len(motif) != 6 or set(motif) - VALID_BASES:
                raise ValueError("invalid 6-mer in zone {}: {}".format(zone, motif))
    return rules


def gc_fraction(sequence):
    return (sequence.count("G") + sequence.count("C")) / len(sequence)


def max_homopolymer(sequence):
    return max(len(match.group(0)) for match in re.finditer(r"(A+|C+|G+|T+)", sequence))


def positions_overlap(first_start, second_start, width=6):
    return not (first_start + width - 1 < second_start or second_start + width - 1 < first_start)


def apply_edits(parent_sequence, edits, coordinate_start):
    bases = list(parent_sequence)
    for relative_start, motif in edits:
        index = relative_start - coordinate_start
        bases[index:index + 6] = list(motif)
    return "".join(bases)


def describe_edits(parent_sequence, edits, coordinate_start):
    descriptions = []
    changed = 0
    for relative_start, motif in edits:
        index = relative_start - coordinate_start
        native = parent_sequence[index:index + 6]
        differences = sum(left != right for left, right in zip(native, motif))
        changed += differences
        descriptions.append(
            "{:+d}..{:+d}:{}>{}({})".format(
                relative_start, relative_start + 5, native, motif, differences
            )
        )
    return ";".join(descriptions), changed


def passes_sequence_constraints(sequence, constraints):
    gc_value = gc_fraction(sequence)
    if not float(constraints["min_gc"]) <= gc_value <= float(constraints["max_gc"]):
        return False
    if max_homopolymer(sequence) > int(constraints["max_homopolymer"]):
        return False
    return not any(site.upper() in sequence for site in constraints.get("forbidden_sites", []))


class DualHeadScorer:
    def __init__(self, device):
        self.device = device
        self.classifier = ModelRunner(MODEL_SPECS["classifier-133bp"], device)
        self.activity = ModelRunner(MODEL_SPECS["activity-133bp"], device)

    def score(self, sequences, batch_size):
        records = [{"sequence": sequence} for sequence in sequences]
        classifier_logits = self.classifier.predict_logits(records, batch_size)
        promoter_scores = torch.softmax(classifier_logits[:, :2], dim=-1)[:, 1]
        activity_standardized = self.activity.predict_logits(records, batch_size).reshape(-1)
        results = []
        for promoter, standardized in zip(promoter_scores.tolist(), activity_standardized.tolist()):
            activity_ln = standardized * ACTIVITY_LABEL_STD + ACTIVITY_LABEL_MEAN
            results.append({
                "promoter_score": float(promoter),
                "activity_standardized": float(standardized),
                "activity_ln": float(activity_ln),
                "activity_linear": max(0.0, math.expm1(activity_ln)),
            })
        return results


def _candidate_states(parent_sequence, beams, rules):
    coordinate_start = int(rules["coordinate_start"])
    constraints = rules["constraints"]
    proposals = []
    seen_sequences = set()
    for state in beams:
        used_starts = [relative_start for relative_start, _ in state["edits"]]
        for slot in rules["slots"]:
            relative_start = int(slot["start"])
            if any(positions_overlap(relative_start, used) for used in used_starts):
                continue
            index = relative_start - coordinate_start
            native = state["sequence"][index:index + 6]
            for motif in rules["motifs_by_zone"][slot["zone"]]:
                if motif == native:
                    continue
                edits = tuple(sorted(state["edits"] + ((relative_start, motif),)))
                sequence = apply_edits(parent_sequence, edits, coordinate_start)
                if sequence in seen_sequences or not passes_sequence_constraints(sequence, constraints):
                    continue
                description, changed = describe_edits(parent_sequence, edits, coordinate_start)
                if changed > int(constraints["max_changed_bp"]):
                    continue
                seen_sequences.add(sequence)
                proposals.append({
                    "sequence": sequence,
                    "edits": edits,
                    "n_edits": len(edits),
                    "n_changed_bp": changed,
                    "edit_description": description,
                })
    return proposals


def optimize_parent(record, scorer, rules, top_k):
    parent_id = record["id"]
    parent_sequence = record["sequence"]
    search = rules["search"]
    constraints = rules["constraints"]
    batch_size = int(search["batch_size"])
    parent_score = scorer.score([parent_sequence], batch_size)[0]
    parent_state = dict(parent_score)
    parent_state.update({
        "sequence": parent_sequence,
        "edits": tuple(),
        "n_edits": 0,
        "n_changed_bp": 0,
        "edit_description": "parent",
    })

    beams = [parent_state]
    retained = []
    for step in range(1, int(search["max_edits"]) + 1):
        proposals = _candidate_states(parent_sequence, beams, rules)
        if not proposals:
            break
        scores = scorer.score([proposal["sequence"] for proposal in proposals], batch_size)
        scored = []
        for proposal, score in zip(proposals, scores):
            proposal.update(score)
            if proposal["promoter_score"] >= float(constraints["min_promoter_score"]):
                scored.append(proposal)
        scored.sort(
            key=lambda row: (row["activity_linear"], row["promoter_score"], row["sequence"]),
            reverse=True,
        )
        beams = scored[:int(search["beam_size"])]
        retained.extend(beams)
        LOGGER.info(
            "%s: step %d generated %d variants; retained %d",
            parent_id, step, len(proposals), len(beams),
        )
        if not beams:
            break

    unique = {}
    for state in retained:
        previous = unique.get(state["sequence"])
        if previous is None or state["activity_linear"] > previous["activity_linear"]:
            unique[state["sequence"]] = state
    retained = sorted(
        unique.values(),
        key=lambda row: (row["activity_linear"], row["promoter_score"], row["sequence"]),
        reverse=True,
    )

    parent_output = {
        "parent_id": parent_id,
        "promoter_score": parent_score["promoter_score"],
        "activity_standardized": parent_score["activity_standardized"],
        "activity_ln": parent_score["activity_ln"],
        "activity_linear": parent_score["activity_linear"],
        "gc_content": gc_fraction(parent_sequence),
        "max_homopolymer": max_homopolymer(parent_sequence),
        "sequence": parent_sequence,
    }
    variants = []
    for rank, state in enumerate(retained, start=1):
        parent_linear = parent_score["activity_linear"]
        variants.append({
            "parent_id": parent_id,
            "variant_id": "{}_opt_{:02d}".format(parent_id, rank),
            "rank": rank,
            "n_edits": state["n_edits"],
            "n_changed_bp": state["n_changed_bp"],
            "promoter_score": state["promoter_score"],
            "parent_activity_linear": parent_linear,
            "activity_standardized": state["activity_standardized"],
            "activity_ln": state["activity_ln"],
            "activity_linear": state["activity_linear"],
            "delta_activity_ln": state["activity_ln"] - parent_score["activity_ln"],
            "delta_activity_linear": state["activity_linear"] - parent_linear,
            "fold_vs_parent": state["activity_linear"] / parent_linear if parent_linear > 0 else "",
            "gc_content": gc_fraction(state["sequence"]),
            "max_homopolymer": max_homopolymer(state["sequence"]),
            "edit_description": state["edit_description"],
            "sequence": state["sequence"],
        })
    return parent_output, variants[:top_k], variants


def optimize_records(records, rules, device, top_k):
    validate_sequences(records, int(rules["sequence_length"]))
    identifiers = [record["id"] for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("input sequence identifiers must be unique")
    scorer = DualHeadScorer(device)
    parents = []
    selected = []
    retained = []
    for record in records:
        LOGGER.info("Optimizing %s", record["id"])
        parent, top_variants, all_variants = optimize_parent(record, scorer, rules, top_k)
        parents.append(parent)
        selected.extend(top_variants)
        retained.extend(all_variants)
    return parents, selected, retained
