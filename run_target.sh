CONFIG="config/single_gpu.yaml"

MODEL_PATH="AntoineBlanot/flan-t5-xxl-classif-2way"
DATA_PATH="/home/chikara/data/fine-tuning/TARGET-MNLI"
RUN_NAME="target"

accelerate launch --config_file $CONFIG train.py \
    --pretrained_model_name_or_path $MODEL_PATH \
    --path $DATA_PATH --bs 32 --seq_length 128 \
    --output_dir "exp/$RUN_NAME" --lr 1e-4 --wd 0 \
    --max_steps 73632 --warmup_steps 7363 --log_steps 100 --eval_steps 1000 --save_steps 1000 \
    --wandb_project "efficient-llm" --wandb_name $RUN_NAME

