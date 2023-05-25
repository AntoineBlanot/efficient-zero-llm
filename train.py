import numpy as np
from torch.utils.data import DataLoader
from evaluate import load
from transformers import HfArgumentParser, BitsAndBytesConfig, T5TokenizerFast, Adafactor, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training

from model.modeling import T5ForClassification
from data import DatasetFromDisk, T5ClassifCollator
from args import TrainingArgs, ModelArgs, DataArgs
from trainer import Trainer

#region Parser
parser = HfArgumentParser([TrainingArgs, ModelArgs, DataArgs])
train_args, model_args, data_args = parser.parse_args_into_dataclasses()
print(train_args ,model_args, data_args, sep='\n')
#endregion


#region Model
device_map = {'': 0}
trainable_layers = ['classif_head']
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_skip_modules=trainable_layers
)
model = T5ForClassification.from_pretrained(model_args.pretrained_model_name_or_path, device_map=device_map, quantization_config=quantization_config)
# Define LoRA Config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=['q', 'v'],
    lora_dropout=0.05,
    bias='none',
    modules_to_save=trainable_layers
)
# prepare int-8 model for training
model = prepare_model_for_int8_training(model, output_embedding_layer_name='', use_gradient_checkpointing=True, layer_norm_names=['layer_norm'])
# add LoRA adaptor
model = get_peft_model(model, lora_config)
print(model)
model.print_trainable_parameters()
#endregion


#region Tokenizer + Data
tokenizer = T5TokenizerFast.from_pretrained(model_args.pretrained_model_name_or_path, model_max_length=data_args.seq_length)
data_collator = T5ClassifCollator(tokenizer, label2id=model.base_model.model.config.label2id)

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
accuracy_metric = load('accuracy')
f1_metric = load('f1')

def compute_metrics(preds, labels):
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    
    acc = accuracy_metric.compute(predictions=preds, references=labels)
    f1 = f1_metric.compute(predictions=preds, references=labels, average='weighted')
    
    metrics = {**acc, **f1}

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