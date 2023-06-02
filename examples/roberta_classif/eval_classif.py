import numpy as np
from torch.utils.data import DataLoader
from evaluate import load
from transformers import HfArgumentParser, BitsAndBytesConfig, RobertaTokenizerFast, RobertaConfig
from peft import PeftModel, PeftConfig

from model.modeling import RobertaForClassification
from data import DatasetFromDisk, RobertaCollator
from args import ModelArgs, DataArgs
from trainer import Trainer

#region Parser
parser = HfArgumentParser([ModelArgs, DataArgs])
model_args, data_args = parser.parse_args_into_dataclasses()
print(model_args, data_args, sep='\n')
#endregion


#region Model
device_map = {'': 0}
trainable_layers = ['classif_head']
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_skip_modules=trainable_layers
)
peft_config = PeftConfig.from_pretrained(model_args.pretrained_model_name_or_path)
base_config = RobertaConfig.from_pretrained(model_args.pretrained_model_name_or_path)

model = RobertaForClassification.from_pretrained(pretrained_model_name_or_path=peft_config.base_model_name_or_path, **base_config.to_diff_dict(), quantization_config=quantization_config, device_map=device_map)
print('Base model loaded')
model = PeftModel.from_pretrained(model, model_args.pretrained_model_name_or_path).eval()
print('Full checkpoint loaded')
print(model)
#endregion


#region Tokenizer + Data
tokenizer = RobertaTokenizerFast.from_pretrained(model_args.pretrained_model_name_or_path, model_max_length=data_args.seq_length)
data_collator = RobertaCollator(tokenizer, label2id=model.base_model.model.config.label2id)

eval_data = DatasetFromDisk(data_args.path + '/dev')
eval_dataloader = DataLoader(eval_data, batch_size=data_args.bs, collate_fn=data_collator)
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
    eval_dataloader=eval_dataloader,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

metrics = trainer.eval()
print(metrics)
