import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from pathlib import Path
from datasets import load_dataset
import os
import importlib

from omegaconf import OmegaConf

from muse_maskgit_pytorch import (
    VQGanVAE,
    MaskGitTrainer,
    MaskGit,
    MaskGitTransformer,
    get_accelerator
)
from muse_maskgit_pytorch.dataset import get_dataset_from_dataroot, ImageTextDataset, split_dataset_into_dataloaders

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

dim = 256
vq_codebook_size = 1024

vae = VQGanVAE(
    dim = dim,
    vq_codebook_size = vq_codebook_size
).cuda()

# model_filename = 'vqgan.1024.model.ckpt'
# config_filename = 'vqgan.1024.config.yml'
# CACHE_PATH = "vae"
# config_path = str(Path(CACHE_PATH) / config_filename)
# model_path = str(Path(CACHE_PATH) / model_filename)
# #model = instantiate_from_config(config["model"])

# state = torch.load(model_path, map_location = 'cpu')['state_dict']
# vae.load_state_dict(state, strict = False)

# vae.save("vae.pt")

vae.load("vae.pt")

"""
python3 train_muse_maskgit.py --num_tokens=1024 --validation_prompt="Product photo of a t-shirt" --resume_from=vae.pt --dim=256 --vq_codebook_size=1024 --train_data_dir=/home/nakano/jellibeans/datasets/img2dataset_v2_dall-e_pa_v1
"""


