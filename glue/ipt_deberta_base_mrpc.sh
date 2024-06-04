export CUDA_VISIBLE_DEVICES='4' 
CUDA_DEVICE=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f1)
BASE_PORT=29520
MASTER_PORT=$(($BASE_PORT + $CUDA_DEVICE)) 
torchrun --nproc_per_node=1 --master_port $MASTER_PORT run_glue.py \
         --wandb_offline 1 \
         --do_train \
         --do_eval \
         --save_total_limit 2 \
         --greater_is_better true \
         --gradient_accumulation_steps 1 \
         --output_dir ./output \
         --overwrite_output_dir \
         --logging_steps 200 \
         --logging_dir ./output/log \
         --evaluation_strategy epoch \
         --save_strategy epoch \
         --warmup_ratio 0.06 \
         --max_grad_norm 0.1 \
         --weight_decay 0.1 \
         --shared_uv 0 \
         --model_name_or_path microsoft/deberta-v3-base \
         --tokenizer_name microsoft/deberta-v3-base \
         --per_device_train_batch_size 64 \
         --max_seq_length 256 \
         --mode elora \
         --lora_r 4 \
         --init_type 1 \
         --d_init_type 94 \
         --seed 42 \
         --task_name mrpc \
         --num_train_epochs 20 \
         --classifier_lr 8e-2 \
         --learning_rate 1e-2 \
         --trainable_uv 1 \
         --disable_tqdm true \
         --freeze_by_epoch 0 \
         --freeze_by_ipt true \
         --lora_dropout 0.0 \
         --load_best_model_at_end true \
