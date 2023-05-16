from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset
from datasets import load_from_disk


class T5Dataset(Dataset):

    def __init__(self, path: str) -> None:
        super().__init__()
        self.data = load_from_disk(path)
        self.n_class = len(self.data.unique('label'))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class T5Collator():

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batched_input_text = [x['input_text'] for x in features]
        batched_target_text = [self.tokenizer.pad_token + x['target_text'] for x in features]
        batched_label = [x['label'] for x in features]

        inputs = self.tokenizer(batched_input_text, padding=True, truncation=True, return_tensors='pt', pad_to_multiple_of=8)
        targets = self.tokenizer(batched_target_text, padding=True, truncation=True, return_tensors='pt', pad_to_multiple_of=8)
        labels = torch.as_tensor(batched_label, dtype=torch.long)

        return dict(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            decoder_input_ids=targets.input_ids,
            decoder_attention_mask=targets.attention_mask,
            labels=labels
        )