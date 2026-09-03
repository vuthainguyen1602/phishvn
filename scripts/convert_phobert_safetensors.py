#!/usr/bin/env python3
"""
convert_phobert_safetensors.py — unblock bin-only Vietnamese checkpoints for transformers 5.x.

transformers 5.x refuses to `torch.load` hub checkpoints that ship only `pytorch_model.bin`
when torch < 2.6 (CVE-2025-32434), and the standard Vietnamese encoders (vinai/phobert-base,
vinai/phobert-base-v2, uitnlp/visobert) ship no safetensors. Rather than force a torch upgrade
into the working environment, this script downloads a public checkpoint once, converts the
weights to safetensors, and assembles a local model directory that `train_content_fusion.py`
picks up automatically (models/<basename>-safetensors).

Run once per model:
  python scripts/convert_phobert_safetensors.py                        # vinai/phobert-base
  python scripts/convert_phobert_safetensors.py vinai/phobert-base-v2
  python scripts/convert_phobert_safetensors.py uitnlp/visobert
"""
from __future__ import annotations
import os
import shutil
import sys

import torch
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors.torch import save_file

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
WEIGHT_SUFFIXES = (".bin", ".h5", ".msgpack", ".safetensors", ".ot", ".onnx")


def convert(repo: str) -> str:
    dst = os.path.join(ROOT, "models", repo.split("/")[-1] + "-safetensors")
    os.makedirs(dst, exist_ok=True)
    for f in list_repo_files(repo):
        if f.endswith(WEIGHT_SUFFIXES) or "/" in f or f.startswith("."):
            continue  # weights handled below; skip nested paths and dotfiles
        shutil.copy(hf_hub_download(repo, f), os.path.join(dst, f))
    binp = hf_hub_download(repo, "pytorch_model.bin")
    sd = torch.load(binp, map_location="cpu", weights_only=True)
    sd = {k: v.contiguous() for k, v in sd.items() if not k.endswith("position_ids")}
    seen = {}  # safetensors rejects aliased storage — clone duplicates
    for k, v in list(sd.items()):
        ptr = v.data_ptr()
        if ptr in seen:
            sd[k] = v.clone()
        seen[ptr] = k
    save_file(sd, os.path.join(dst, "model.safetensors"))
    print(f"[+] {dst}: {sorted(os.listdir(dst))}")
    return dst


if __name__ == "__main__":
    convert(sys.argv[1] if len(sys.argv) > 1 else "vinai/phobert-base")
