# TRAIN
CONFIG="config/single_gpu.yaml"
MODEL_PATH="AntoineBlanot/roberta-nli"
DATA_PATH="/home/chikara/ws/datasets/examples/NLI"

RUN_NAME="roberta-nli"

accelerate launch --config_file $CONFIG train_classif.py \
  --pretrained_model_name_or_path $MODEL_PATH \
  --path $DATA_PATH --bs 128 --seq_length 128 \
  --output_dir "exp/$RUN_NAME" --lr 1e-4 --wd 0 \
  --max_steps 7544 --warmup_steps 755 --log_steps 100 --eval_steps 1000 --save_steps 1000 \
  --wandb_project "efficient-llm" --wandb_name $RUN_NAME

# EVAL
CONFIG="config/single_gpu.yaml"
MODEL_PATH="/home/chikara/ws/efficient-llm/exp/roberta-nli/checkpoint-7000"
DATA_PATH="/home/chikara/ws/datasets/examples/NLI"

accelerate launch --config_file $CONFIG eval_classif.py \
  --pretrained_model_name_or_path $MODEL_PATH \
  --path $DATA_PATH --bs 32 --seq_length 512
