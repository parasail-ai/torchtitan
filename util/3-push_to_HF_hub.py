from transformers import AutoTokenizer, LlamaForCausalLM
import torch

# This script push the local HF model repo to your HF hub
# This is done by first loading the model into transformers,
# and then push both the model and the tokenizer to the hub

# !!! Set your target HF hub repo id !!!
hub_id = "parasail-ai/MicroLlama3_slimpajama_starcoderdata_ckpt_step-800k"

# Then just run
# `python 3-push_to_HF_hub.py`

# In principle you don't have to change anything below
output_dir = "./to_hf_output"
model = LlamaForCausalLM.from_pretrained(output_dir, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(output_dir)

print("Pushing model to hub")
model.push_to_hub(hub_id)
print("Pushing tokenizer to hub")
tokenizer.push_to_hub(hub_id)
