# Efficient LLM

## Introduction

Training and evaluation scripts for Large Language Models (LLMs) in an efficient way. You can train models with billions of parameters with few resources.

## Installation
Please install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) (miniconda recommended) for the environment manager.<br>

Once conda is installed, you can create and activate the environment using the following commads.

```
conda env create -f efficient-llm.yml
conda activate efficient-llm
```

## Trained models
Model checkpoints can be find in this repo: https://github.com/AntoineBlanot/zero-nlp

Reported scores are accuracy on the validation set.
| Model name | Size | MNLI (m) | MNLI (mm) | SNLI | SciTail |
|:----------:|:----:|:--------:|:---------:|:----:|:-------:|
| 3way-nli-mixture | 5B | 0.923 | 0.922 | 0.942 | 0.966 |

## 1. Training
To train your model efficiently, you can run a simple command and pass arguments throught it like the config file, model path and data path. You can also pass different hyperparameters like the batch size, learning rate, weight decay and many other.

You can check all the arguments supported right now in the [args.py](args.py) file.

Here is how a training script looks like:
```
CONFIG="config/single_gpu.yaml"
MODEL_PATH="AntoineBlanot/flan-t5-xxl-classif-3way"
DATA_PATH=...

RUN_NAME="3way-nli-mixture"

accelerate launch --config_file $CONFIG train.py \
  --pretrained_model_name_or_path $MODEL_PATH \
  --path $DATA_PATH --bs 32 --seq_length 128 \
  --output_dir "exp/$RUN_NAME" --lr 1e-4 --wd 0 \
  --max_steps 30178 --warmup_steps 3018 --log_steps 500 --eval_steps 5000 --save_steps 5000 \
  --wandb_project "efficient-llm" --wandb_name $RUN_NAME

```

## 2. Evaluation
To evaluate your model you need a similar command. The model path is the path to the checkpoint you want to evaluate on.

Here is how an evaluation script looks like:
```
CONFIG="config/single_gpu.yaml"
MODEL_PATH=".../checkpoint-..."
DATA_PATH=...

accelerate launch --config_file $CONFIG eval.py \
  --pretrained_model_name_or_path $MODEL_PATH \
  --path $DATA_PATH --bs 32 --seq_length 128

```

## 3. Inference
For zero-shot inference, please refer to another of my repo: https://github.com/AntoineBlanot/zero-nlp

Simple inference is not implemented for now...
