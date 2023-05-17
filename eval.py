
from torch.utils.data import DataLoader
from evaluate import load
from transformers import HfArgumentParser, AutoTokenizer, AutoConfig
from peft import PeftModel, PeftConfig

from model.modeling import T5ForClassification
from data import T5Dataset, T5Collator
from args import ModelArgs, DataArgs
from trainer import Trainer

#region Parser
parser = HfArgumentParser([ModelArgs, DataArgs])
model_args, data_args = parser.parse_args_into_dataclasses()
print(model_args, data_args, sep='\n')
#endregion


#region Model
peft_config = PeftConfig.from_pretrained(model_args.pretrained_model_name_or_path)
base_config = AutoConfig.from_pretrained(model_args.pretrained_model_name_or_path)

model = T5ForClassification.from_pretrained(pretrained_model_name_or_path=peft_config.base_model_name_or_path, **base_config.to_diff_dict(), load_in_8bit=True, device_map={'': 0})
model = PeftModel.from_pretrained(model, model_args.pretrained_model_name_or_path, device_map={'': 0})
model.eval()
#endregion


#region Tokenizer + Data
tokenizer = AutoTokenizer.from_pretrained(model_args.pretrained_model_name_or_path)
data_collator = T5Collator(tokenizer)

eval_data = T5Dataset(data_args.path + "/dev")
eval_dataloader = DataLoader(eval_data, batch_size=data_args.bs, collate_fn=data_collator)
#endregion


#region Metrics
accuracy_metric = load("accuracy")
f1_metric = load("f1")

def compute_metrics(preds, labels):
    acc = accuracy_metric.compute(predictions=preds, references=labels)
    f1 = f1_metric.compute(predictions=preds, references=labels, average='macro')
    
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