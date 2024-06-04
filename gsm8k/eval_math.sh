CUDA_VISIBLE_DEVICES=0 python evaluate.py --model LLaMA-7B --adapter aflora \
  --dataset gsm8k --base_model 'yahma/llama-7b-hf' \
  --lora_weights 'trained_models/llama-7b-math10k/xxx'