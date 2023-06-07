CONFIG="config/single_gpu.yaml"

MODEL_PATH="AntoineBlanot/t5-uNLU-large"
DATA_PATH="/home/chikara/ws/datasets/examples/MNLI"
RUN_NAME="large-24-mnli"

accelerate launch --config_file $CONFIG train.py \
    --pretrained_model_name_or_path $MODEL_PATH \
    --path $DATA_PATH --bs 32 --seq_length 128 \
    --output_dir "exp/$RUN_NAME" --lr 1e-4 --wd 0 \
    --max_steps 24544 --warmup_steps 2454 --log_steps 100 --eval_steps 1000 --save_steps 1000 \
    --wandb_project "uNLU" --wandb_name $RUN_NAME

# # EVAL
# CONFIG="config/single_gpu.yaml"
# MODEL_PATH="/home/chikara/ws/zero-nlp/model/best-seq-classif"
# DATA_PATH="/home/chikara/ws/datasets/examples/MNLI"

# accelerate launch --config_file $CONFIG eval_classif.py \
#   --pretrained_model_name_or_path $MODEL_PATH \
#   --path $DATA_PATH --bs 32 --seq_length 512
