# TRAIN
CONFIG="config/single_gpu.yaml"
MODEL_PATH="AntoineBlanot/roberta-span-detection"
DATA_PATH="/home/chikara/ws/datasets/examples/SPAN-DETECTION"

RUN_NAME="roberta-span"

accelerate launch --config_file $CONFIG train_token.py \
  --pretrained_model_name_or_path $MODEL_PATH \
  --path $DATA_PATH --bs 32 --seq_length 128 \
  --output_dir "exp/$RUN_NAME" --lr 1e-4 --wd 0 \
  --max_steps 36084 --warmup_steps 3609 --log_steps 100 --eval_steps 1000 --save_steps 1000 \
  --wandb_project "efficient-llm" --wandb_name $RUN_NAME


# EVAL
CONFIG="config/single_gpu.yaml"
MODEL_PATH="/home/chikara/ws/efficient-llm/exp/roberta-span/checkpoint-36000"
DATA_PATH="/home/chikara/ws/datasets/examples/SPAN-DETECTION"

accelerate launch --config_file $CONFIG eval_token.py \
  --pretrained_model_name_or_path $MODEL_PATH \
  --path $DATA_PATH --bs 32 --seq_length 128

  