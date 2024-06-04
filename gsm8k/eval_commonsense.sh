export BASE_MODEL=yahma/llama-7b-hf
weights_path='xxx'
CUDA_VISIBLE_DEVICES=0 python commonsense_evaluate.py --model LLaMA-7B --base_model 'yahma/llama-7b-hf' --data 'boolq' \
  --batch_size 4 --adapter dora --lora_weights $weights_path