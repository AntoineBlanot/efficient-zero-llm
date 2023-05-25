from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from evaluate import load
from transformers import HfArgumentParser, BitsAndBytesConfig, Adafactor, RobertaTokenizerFast, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training

from model.modeling import RobertaForTokenClassification
from data import DatasetFromDisk, RobertaTokenCollator
from args import TrainingArgs, ModelArgs, DataArgs
from trainer import Trainer

#region Parser
parser = HfArgumentParser([TrainingArgs, ModelArgs, DataArgs])
train_args, model_args, data_args = parser.parse_args_into_dataclasses()
print(train_args ,model_args, data_args, sep='\n')
#endregion


#region Model
device_map = {'': 0}
trainable_layers = ['token_head']
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_skip_modules=trainable_layers
)
model = RobertaForTokenClassification.from_pretrained(model_args.pretrained_model_name_or_path, device_map=device_map, quantization_config=quantization_config)
# Define LoRA Config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=['query', 'value'],
    lora_dropout=0.05,
    bias='none',
    modules_to_save=trainable_layers
)
# prepare int-8 model for training
model = prepare_model_for_int8_training(model)
# add LoRA adaptor
model = get_peft_model(model, lora_config)
print(model)
model.print_trainable_parameters()
#endregion


#region Tokenizer + Data
tokenizer = RobertaTokenizerFast.from_pretrained(model_args.pretrained_model_name_or_path, model_max_length=data_args.seq_length, add_prefix_space=True)
data_collator = RobertaTokenCollator(tokenizer, label2id=model.base_model.model.config.label2id)

train_data = DatasetFromDisk(data_args.path + '/train')
eval_data = DatasetFromDisk(data_args.path + '/dev')

train_dataloader = DataLoader(train_data, batch_size=data_args.bs, collate_fn=data_collator)
eval_dataloader = DataLoader(eval_data, batch_size=data_args.bs, collate_fn=data_collator)
#endregion


#region Optimizer + LR Scheduler
optimizer = Adafactor(
    model.parameters(),
    lr=train_args.lr,
    eps=(1e-30, 1e-3),
    clip_threshold=1.0,
    decay_rate=-0.8,
    beta1=None,
    weight_decay=train_args.wd,
    relative_step=False,
    scale_parameter=False,
    warmup_init=False,
)  
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=train_args.warmup_steps, num_training_steps=train_args.max_steps)
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
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    tokenizer=tokenizer,
    optimizer=optimizer,
    scheduler=scheduler,
    train_args=train_args,
    compute_metrics=compute_metrics
)

metrics = trainer.train()
print(metrics)