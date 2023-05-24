from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset
from datasets import load_from_disk


class DatasetFromDisk(Dataset):
    '''
    Dataset class that loads data locally.
    '''
    
    def __init__(self, path: str) -> None:
        super().__init__()
        self.data = load_from_disk(path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class T5ClassifCollator():
    '''
    Data collator for `T5` models for `sequence classification` task

    Required features in the dataset:
        - `input_text`: text to feed to the encoder (str)
        - `target_text`: text to feed to the decoder (str)
        - `label`: target class index (int)
    '''

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batched_input_text = [x['input_text'] for x in features]
        batched_target_text = [self.tokenizer.pad_token + x['target_text'] for x in features]
        batched_label = [x['label'] for x in features]

        inputs = self.tokenizer(batched_input_text, padding=True, truncation=True, pad_to_multiple_of=8, return_tensors='pt')
        targets = self.tokenizer(batched_target_text, padding=True, truncation=True, pad_to_multiple_of=8, return_tensors='pt')
        labels = torch.as_tensor(batched_label, dtype=torch.long)

        return dict(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            decoder_input_ids=targets.input_ids,
            decoder_attention_mask=targets.attention_mask,
            labels=labels
        )


class RobertaTokenCollator():
    '''
    Data collator for `BERT` models for `token classification` task

    Required features in the dataset:
        - `tokens`: pre-tokenized (split into words) text (list of str)
        - `ner_tags`: target tags for each pre-tokenized token (list of int)
    '''

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.preprocess_fn = self.tokenize_and_align_labels

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batched_tokens = [x['tokens'] for x in features]
        batched_ner_tags = [x['ner_tags'] for x in features]

        examples_dict = dict(
            tokens=batched_tokens,
            ner_tags=batched_ner_tags
        )

        inputs = self.preprocess_fn(examples=examples_dict)

        return inputs
    
    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(examples['tokens'], is_split_into_words=True, padding=True, truncation=True, pad_to_multiple_of=8, return_tensors='pt')

        labels = []
        for i, label in enumerate(examples['ner_tags']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs['labels'] = torch.as_tensor(labels, dtype=torch.long)

        return tokenized_inputs