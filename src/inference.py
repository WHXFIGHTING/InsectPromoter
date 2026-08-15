import math

import torch

from .config import ACTIVITY_LABEL_MEAN, ACTIVITY_LABEL_STD, MODEL_SPECS
from .model import predict_logits


def add_classification_scores(records, logits, sequence_length):
    probabilities = torch.softmax(logits, dim=-1)[:, 1].tolist()
    column = "promoter_score_300bp" if sequence_length == 300 else "promoter_score_133bp"
    for record, probability in zip(records, probabilities):
        record[column] = probability


def add_activity_scores(records, logits):
    for record, value in zip(records, logits.reshape(-1).tolist()):
        activity_ln = value * ACTIVITY_LABEL_STD + ACTIVITY_LABEL_MEAN
        record["activity_standardized"] = value
        record["activity_ln"] = activity_ln
        record["activity_linear"] = max(0.0, math.expm1(activity_ln))


def add_predictions(records, model_name, device, batch_size):
    spec = MODEL_SPECS[model_name]
    logits = predict_logits(records, spec, device, batch_size)
    if spec["task"] == "classification":
        add_classification_scores(records, logits, spec["length"])
    else:
        add_activity_scores(records, logits)
