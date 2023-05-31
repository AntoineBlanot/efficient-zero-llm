from transformers import BitsAndBytesConfig, RobertaTokenizerFast, RobertaConfig
from peft import PeftModel, PeftConfig

from model.modeling import RobertaForTokenClassification

MODEL_PATH = "/home/chikara/ws/efficient-llm/exp/best-token-classif"
TOKENS = [ "The", "following", "were", "among", "Friday", "'s", "offerings", "and", "pricings", "in", "the", "U.S.", "and", "non-U.S.", "capital", "markets", ",", "with", "terms", "and", "syndicate", "manager", ",", "as", "compiled", "by", "Dow", "Jones", "Capital", "Markets", "Report", ":" ]

#region Model
device_map = {'': 0}
trainable_layers = ['token_head']
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_skip_modules=trainable_layers
)
peft_config = PeftConfig.from_pretrained(MODEL_PATH)
base_config = RobertaConfig.from_pretrained(MODEL_PATH)

model = RobertaForTokenClassification.from_pretrained(pretrained_model_name_or_path=peft_config.base_model_name_or_path, **base_config.to_diff_dict(), quantization_config=quantization_config, device_map=device_map)
print('Base model loaded')
model = PeftModel.from_pretrained(model, MODEL_PATH, device_map=device_map)
print('Full checkpoint loaded')
model.eval()
#endregion

#region Tokenizer + Data
tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_PATH, model_max_length=512, add_prefix_space=True)
#endregion

inputs = tokenizer(TOKENS, is_split_into_words=True, return_tensors='pt')
logits = model(**inputs)['logits'][0]

processed_inputs = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
preds = [model.base_model.model.config.id2label[i] for i in logits.argmax(-1).tolist()]

spans, current_span = [], []
tags, current_tag = [], []
for i, p in zip(processed_inputs, preds):
    if p != 'O':
        if p.startswith('B'):
            current_span.append(i.replace('Ġ', ''))
            current_tag.append(p)
        else:
            current_span.append(i.replace('Ġ', ' '))
            current_tag.append(p)
    else:
        if current_span != []:
            spans.append(current_span)
            tags.append(current_tag)
        current_span, current_tag = [], []

print('----- RESULTS -----')
print('Input tokens: {}'.format(TOKENS))
print('Possible tags: {}'.format(model.base_model.model.config.label2id.keys()))
print('Detected tags: {}'.format(tags))
print('Detected spans: {}'.format(spans))
print('Detected texts: {}'.format([''.join(x) for x in spans]))
