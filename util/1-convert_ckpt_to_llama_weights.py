import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
import os
import shutil

# The original llama model repo structure was uploaded with the following files:
# "consolidated.00.pth", "params.json", "tokenizer.model"
# See for example https://huggingface.co/meta-llama/Meta-Llama-3-70B/tree/main/original
# We use this script to prepare these file from the torch DCP files.

# !!! Set these parameters before you run the script !!!
model_arch_name = "Micro"
dcp_checkpoint_dir = "/nvme0n1/ben/torchtitan/outputs_MicroLlama3/checkpoint/step-800000"
tokenizer_path = "/nvme0n1/ben/torchtitan/torchtitan/datasets/tokenizer/original/tokenizer.model"

# In principle you don't need to change anything below
# Just run `python 1-convert_ckpt_to_llama_weights.py`
# A directory "to_HF_input" will be created, which will be used for the next conversion

# Model arch settings
# Here I hard-coded the arch params for the three models "Tiny", "Micro", and "Nano" LLama3
# You need to add a definition yourself if you are converting a model you defined on your own
params_json = {}
params_json["Tiny"] = \
"""
{
    "dim": 2048,
    "n_heads": 32,
    "n_kv_heads": 8,
    "n_layers": 22,
    "multiple_of": 1024,
    "norm_eps": 1e-05,
    "vocab_size": 128256,
    "rope_theta": 500000.0
}
"""
params_json["Micro"] = \
"""
{
    "dim": 1024,
    "n_heads": 16,
    "n_kv_heads": 8,
    "n_layers": 16,
    "multiple_of": 1024,
    "norm_eps": 1e-05,
    "vocab_size": 128256,
    "rope_theta": 500000.0
}
"""
params_json["Nano"] = \
"""
{
    "dim": 512,
    "n_heads": 8,
    "n_kv_heads": 4,
    "n_layers": 8,
    "multiple_of": 512,
    "norm_eps": 1e-05,
    "vocab_size": 128256,
    "rope_theta": 500000.0
}
"""

# Running the conversion
torch_save_path = "tmp.pt"

to_hf_input = "./to_hf_input/"

if os.path.exists(to_hf_input):
    shutil.rmtree(to_hf_input)
os.mkdir(to_hf_input)

print("Converting dcp to torch save")
dcp_to_torch_save(dcp_checkpoint_dir, torch_save_path)

print("Converting torch save to a consolidated tensor")
model = torch.load(torch_save_path)
model = model["model"]
torch.save(model, to_hf_input+"consolidated.00.pth")
os.remove(torch_save_path)

print("Linking tokenizer")
os.symlink(tokenizer_path, to_hf_input+"tokenizer.model")

print("Writing params.json")
with open(to_hf_input + "params.json", "w") as f:
    f.write(params_json[model_arch_name])

print("Done.")
