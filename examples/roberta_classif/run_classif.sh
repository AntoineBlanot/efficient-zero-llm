# TRAIN
CONFIG="config/single_gpu.yaml"
MODEL_PATH="AntoineBlanot/roberta-nli"
DATA_PATH="/home/chikara/ws/datasets/examples/NLI"

RUN_NAME="roberta-nli"

accelerate launch --config_file $CONFIG train_classif.py \
  --pretrained_model_name_or_path $MODEL_PATH \
  --path $DATA_PATH --bs 128 --seq_length 128 \
  --output_dir "exp/$RUN_NAME" --lr 1e-4 --wd 0 \
  --max_steps 75450 --warmup_steps 7545 --log_steps 500 --eval_steps 5000 --save_steps 5000 \
  --wandb_project "efficient-llm" --wandb_name $RUN_NAME


# # EVAL
# CONFIG="config/single_gpu.yaml"
# MODEL_PATH="..."
# DATA_PATH="/home/chikara/ws/datasets/examples/NLI"

# accelerate launch --config_file $CONFIG eval_classif.py \
#   --pretrained_model_name_or_path $MODEL_PATH \
#   --path $DATA_PATH --bs 32 --seq_length 512
