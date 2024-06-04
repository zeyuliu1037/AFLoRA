from safetensors.torch import load_file
import torch

if __name__ == "__main__":
    filename = "trained_models/llama-7b-math/lora_r32_0.0003_0313221445/adapter_model.safetensors"
    path_name = filename.split("/")[:-1]
    path_name = "/".join(path_name)
    pt_state_dict = load_file(
    filename, device="cpu")
    torch.save(pt_state_dict, path_name + "/adapter_model.bin")
