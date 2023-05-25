from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from evaluate import load
from transformers import HfArgumentParser, BitsAndBytesConfig, RobertaTokenizerFast, RobertaConfig
from peft import PeftModel, PeftConfig

from model.modeling import RobertaForTokenClassification
from data import DatasetFromDisk, RobertaTokenCollator
from args import ModelArgs, DataArgs
from trainer import Trainer

#region Parser
parser = HfArgumentParser([ModelArgs, DataArgs])
model_args, data_args = parser.parse_args_into_dataclasses()
print(model_args, data_args, sep='\n')
#endregion


#region Model
device_map = {'': 0}
trainable_layers = ['token_head']
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_skip_modules=trainable_layers
)
peft_config = PeftConfig.from_pretrained(model_args.pretrained_model_name_or_path)
base_config = RobertaConfig.from_pretrained(model_args.pretrained_model_name_or_path)

model = RobertaForTokenClassification.from_pretrained(pretrained_model_name_or_path=peft_config.base_model_name_or_path, **base_config.to_diff_dict(), quantization_config=quantization_config, device_map=device_map)
print('Base model loaded')
model = PeftModel.from_pretrained(model, model_args.pretrained_model_name_or_path, device_map=device_map)
print('Full checkpoint loaded')
model.eval()
#endregion


#region Tokenizer + Data
tokenizer = RobertaTokenizerFast.from_pretrained(model_args.pretrained_model_name_or_path, model_max_length=data_args.seq_length, add_prefix_space=True)
data_collator = RobertaTokenCollator(tokenizer, label2id=model.base_model.model.config.label2id)

eval_data = DatasetFromDisk(data_args.path + '/dev')
eval_dataloader = DataLoader(eval_data, batch_size=data_args.bs, collate_fn=data_collator)
#endregion


#region Metrics
seqeval = load('seqeval')
label_name_list = [x for x in model.base_model.model.config.label2id.keys()]

def compute_metrics(preds_list: List[torch.Tensor], labels_list: List[torch.Tensor]):
    '''
    Compute metrics for token classification task.

        1. Pad the predictions and labels for concatenation
        2. Remove tokens that do not have to be classified (class -100)
    '''
    preds = np.concatenate([
        F.pad(p, (0, data_args.seq_length - p.shape[1]), value=-100)
        for p in preds_list
    ])
    labels = np.concatenate([
        F.pad(l, (0, data_args.seq_length - l.shape[1]), value=-100)
        for l in labels_list
    ])

    true_predictions = [
        [label_name_list[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(preds, labels)
    ]
    true_labels = [
        [label_name_list[l] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(preds, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)

    metrics = {'accuracy': results['overall_accuracy'], 'f1': results['overall_f1']}

    return metrics
#endregion


trainer = Trainer(
    model=model,
    eval_dataloader=eval_dataloader,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

metrics = trainer.eval()
print(metrics)