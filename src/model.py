import torch
from transformers import AutoTokenizer

from .config import MAX_LENGTH, TOKENIZER_PATH
from .dnabert2.bert_layers import BertForSequenceClassification
from .dnabert2.configuration_bert import BertConfig
from .preprocessing import prepare_sequences


def load_model(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    config = BertConfig.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(
        model_path, config=config
    ).to(device)
    return tokenizer, model.eval()


class ModelRunner:
    def __init__(self, spec, device):
        self.spec = spec
        self.device = device
        self.tokenizer, self.model = load_model(spec["path"], device)

    def predict_logits(self, records, batch_size):
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(records), batch_size):
                batch_records = records[start : start + batch_size]
                sequences = prepare_sequences(batch_records, self.spec["preprocess"])
                encoded = self.tokenizer(
                    sequences,
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt",
                )
                encoded = {name: value.to(self.device) for name, value in encoded.items()}
                outputs.append(self.model(**encoded).logits.detach().cpu())
        return torch.cat(outputs, dim=0)


def predict_logits(records, spec, device, batch_size):
    runner = ModelRunner(spec, device)
    return runner.predict_logits(records, batch_size)
