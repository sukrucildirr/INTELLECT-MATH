#!/bin/bash

NUM_GPUS=8
dataset_mixer_list=${dataset_mixer_list:-"PrimeIntellect/Notus-7B-SFT-Data 1.0"}

export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))

echo "Number of GPUs: $NUM_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --model_name_or_path Qwen/Qwen2.5-Math-7B \
    --model_revision main \
    --use_flash_attn true \
    --tokenizer_name Qwen/Qwen2.5-Math-7B-Instruct \
    --use_slow_tokenizer true \
    --dataset_mixer_list $dataset_mixer_list \
    --preprocessing_num_workers 128 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5.0e-06 \
    --max_seq_length 5000 \
    --gradient_checkpointing \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 5 \
    --checkpointing_steps 300 \
    --output_dir output/notus_7b_sft \
    --with_tracking true \
    --report_to wandb \
    --logging_steps 1 \
    --dataset_mix_dir output/notus_7b_sft \
    --save_hf_checkpoint true