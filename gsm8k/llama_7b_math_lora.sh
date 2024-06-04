CUDA_VISIBLE_DEVICES=0 python finetune.py \
  --base_model 'yahma/llama-7b-hf' \
  --data_path 'ft-training_set/math_10k.json' \
  --output_dir './trained_models/llama-7b-math/' \
  --batch_size 1  --micro_batch_size 4 \
  --num_epochs 3   --learning_rate 3e-4 \
  --cutoff_len 256   --val_set_size 120 \
  --eval_step 80 --save_step 80  \
  --adapter_name aflora \
  --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
  --lora_r 32 --lora_alpha 64 \
  --wandb_project 'llama-7b-math10k'