
### Before training, first download Llama 3 family's tokenizer

`python torchtitan/datasets/download_tokenizer.py --repo_id meta-llama/Meta-Llama-3-8B --tokenizer_path "original"`

The tokenizer is now stored in `torchtitan/torchtitan/datasets/tokenizer/original/tokenizer.model`.

### Custom model configurations

I have trained three custom models: "Tiny", "Micro", and "Nano". The Tiny one follows a similar architecture of the original TinyLlama. To add more custom models, simply edit `torchtitan/torchtitan/models/llama/__init__.py`.

### Training dataset

The original torchtitan implementation only uses `allenai/c4` dataset. I have edited `torchtitan/torchtitan/datasets/hf_datasets.py` to hard-code the combined usage of SlimPajama and Starcoderdata datasets, which are downloaded and stored locally. I used data streaming because these datasets are huge. I also used interleave_datasets to make sure that the model sees both datasets during training. 

One caveat is that there seems to be an issue for streaming the SlimPajama dataset directly from the HF hub, which is why I eventually used the predownloaded datasets to stream.

### Training configs

The training configs are stored in `torchtitan/train_configs`. My custom configs are `llama3_tiny.toml`, `llama3_micro.toml`, and `llama3_nano.toml21`. In the config files you can specify the training datasets. I trained these configs with the dataset options "c4" (torchtitan default) and "slimpajama+starcoderdata" hard-coded. Note that in order to enable continued training from an interrupted run, you must make sure `model_weights_only = false`, such that the training progress is also included in the output of every checkpoint save.

### Run training

In `run_llama_train.sh`, modify the parameters `NGPU` and `CONFIG_FILE`, then run `./run_llama_train.sh`.

### Monitor training progress

Port the tensorboard to local at 6006 when connecting using ssh: `ssh -L 6006:127.0.0.1:6006 ben@sakura-h100-3.parasail-dev.com`. Then run `tensorboard --logdir outputs/tb` on your remote machine. Then view the board at `http://localhost:6006/` on your local machine.

### Upload checkpoints to Hugging Face

This part is a bit messy. First choose a checkpoint, for example "outputs/checkpoint/step-50000" (you can set the checkpoint frequency using the `interval` parameter in the training configs), you will see files like `__0_0.distcp  __1_0.distcp  .metadata`. These are the model weights in the torch [distributed checkpoint (DCP)](https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html) format. We will follow the process below to convert the DCP step-by-step to an HF repo.

`DCP -> Torch save format -> original Llama repo structure -> HF repo structure -> Uploading`

I have provided 3 scripts in `torchtitan/util` to accomplish all these conversions.

 - 1-convert_ckpt_to_llama_weights.py
 - 2-convert_llama_weights_to_hf.py
 - 3-push_to_HF_hub.py

 You need to set a few parameters in scripts and 1 and 3.

### Detailed explanation of conversions

#### DCP -> Torch save
Use [`torch.distributed.checkpoint.format_utils.dcp_to_torch_save()`](https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.format_utils.torch_save_to_dcp).

#### Torch save -> original Llama repo structure

The original Llama repo structure contains three files "consolidated.00.pth", "params.json", "tokenizer.model".

- "consolidated.00.pth" - The torch save file for only the model weights. Note that since your converted torch save file contains training progress data, you need to extract only the model weights by loading the whole torch save, get the weights, and torch.save() again, like `torch.save(torch.load(converted_torch_save_path)['model'], 'consolidated.00.pth')`

- "params.json" - You need to write this file manually. But it's easy by inspecting the [original Llama 3 param file](https://huggingface.co/meta-llama/Meta-Llama-3-70B/blob/main/original/params.json) and your training config file. I have hard coded the param.json files for my custom Tiny, Micro, and Nano models in script 1.

- "tokenizer.model" - Exactly the same file as that in any Llama 3 model repo. Just link it.

#### Original Llama repo structure -> HF repo

Use script ["convert_llama_weights_to_hf.py" from Transformers](https://github.com/huggingface/transformers/blob/3345ae733b6f4aeb7204a0f3e646a3cdbaad0023/src/transformers/models/llama/convert_llama_weights_to_hf.py). I have provided a modified version for this script in script 2.

#### HF repo -> Uploading to HF hub

The idea is to first load the HF repo into transformers and upload it to the hub using `.push_to_hub()`. Both the model and tokenizer needs to be pushed to make the remote repo directly deployable.

#### Steps after uploading

Navigate to HF hub and make the uploaded repo private (it's public by default), also add it to an appropriate collection.

### Version incompatibility for `freqs_cis` errors

I used different versions of torchtitan to train on Sakura 3 and 4. I manually turned off `freq_cis` layer export on Sakura 3, while the current main branch of torchtitan and the parasail fork (which is used on Sakura 4) will expect to see this layer in the checkpoint. Therefore, if you try to continue training of a checkpoint from Sakura 3 on Sakura 4, you may encounter
`RuntimeError: Missing key in checkpoint state_dict: model.freqs_cis.` To bypass this, you can manually [turn off the persistance of freq_cis layers](https://github.com/parasail-ai/torchtitan/blob/7052ffb9e265e606392d6c81f4887e3e982f4dbf/torchtitan/models/llama/model.py#L367). This way torchtitan won't look for `freqs_cis` layers in your checkpoint files.

#### Note 
`freqs_cis` is a CONSTANT layer in the llama models. They can be [precomputed](https://github.com/pytorch/torchtitan/blob/b0ed7f075921357b01e28fddc6d90a2cc410bab3/torchtitan/models/llama/model.py#L408) before model deployement, and they only depends on the model architecture – namely they don't need to be trained. Since they are not trained weights, in principle you don't need to save them to checkpoints. So I removed them from the exported checkpoints for experimental purposes on Sakura 3. But the current [torchtitan implementation](https://github.com/parasail-ai/torchtitan/blob/7052ffb9e265e606392d6c81f4887e3e982f4dbf/torchtitan/models/llama/model.py#L360) (and our parasail fork, which is used on sakura 4) still has it turned on.