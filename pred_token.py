from transformers import BitsAndBytesConfig, RobertaTokenizerFast, RobertaConfig
from peft import PeftModel, PeftConfig

from model.modeling import RobertaForTokenClassification

MODEL_PATH = "/home/chikara/ws/efficient-llm/exp/best-token-classif"
TOKENS = [ "$", "150", "million", "of", "9", "%", "debentures", "due", "Oct.", "15", ",", "2019", ",", "priced", "at", "99.943", "to", "yield", "9.008", "%", "." ]
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

preds = [model.base_model.model.config.id2label[i] for i in logits.argmax(-1).tolist()]

def postprocess(inputs, preds):
    res_dict = {'tags': [], 'ids': []}
    current_tag, current_id = None, None

    for i, p in enumerate(preds):
        id = inputs.input_ids[0][i].item()   # current id
        if p.startswith('B'):
            res_dict['tags'].append(current_tag) if current_tag is not None else None
            res_dict['ids'].append(current_id) if current_id is not None else None
            current_tag, current_id = [p], [id]
        if p.startswith('I'):
            current_tag.append(p)
            current_id.append(id)

    res_dict['tags'].append(current_tag) if current_tag is not None else None
    res_dict['ids'].append(current_id) if current_id is not None else None
    
    return res_dict

res = postprocess(inputs, preds)
res['tokens'] = [tokenizer.convert_ids_to_tokens(i) for i in res['ids']]

print('----- RESULTS -----')
print('Input tokens: {}'.format(TOKENS))
print('Possible tags: {}'.format(model.base_model.model.config.label2id.keys()))
print('Detected tags: {}'.format(res['tags']))
print('Detected ids: {}'.format(res['ids']))
print('Detected tokens: {}'.format(res['tokens']))
print('Detected texts: {}'.format([''.join(toks).replace('Ġ', ' ').strip() for toks in res['tokens']]))
