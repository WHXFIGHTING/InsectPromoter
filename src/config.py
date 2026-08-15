from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = ROOT / "tokenizer"
MODEL_SPECS = {
    "classifier-300bp": {
        "path": ROOT / "models/classifier_300bp",
        "length": 300,
        "preprocess": "raw",
        "task": "classification",
    },
    "classifier-133bp": {
        "path": ROOT / "models/classifier_133bp",
        "length": 133,
        "preprocess": "kmer6",
        "task": "classification",
    },
    "activity-133bp": {
        "path": ROOT / "models/activity_133bp",
        "length": 133,
        "preprocess": "kmer6",
        "task": "regression",
    },
}
MODEL_NAMES = tuple(MODEL_SPECS)
ACTIVITY_LABEL_MEAN = 0.38278340597199106
ACTIVITY_LABEL_STD = 0.8085157980161298
MAX_LENGTH = 512
