export CUDA_VISIBLE_DEVICES='0' 
CUDA_DEVICE=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f1)
BASE_PORT=29520
MASTER_PORT=$(($BASE_PORT + $CUDA_DEVICE))  
WORLD_SIZE=1
python finetune.py \
  --base_model 'yahma/llama-7b-hf' \
  --data_path 'ft-training_set/math_10k.json' \
  --output_dir './trained_models/llama-7b-math10k/' \
  --batch_size 1  --micro_batch_size 1 \
  --num_epochs 3   --learning_rate 2e-3 \
  --cutoff_len 256   --val_set_size 120 \
  --eval_step 80 --save_step 80  \
  --adapter_name aflora \
  --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
  --lora_r 32 --lora_alpha 64 \
  --wandb_project 'llama-7b-math10k'
