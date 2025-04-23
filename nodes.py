import argparse
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import soundfile as sf
import torch

from dia.model import Dia


class LoadDiaModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": "nari-labs/Dia-1.6B",
                }),
                "device": ("STRING", {
                    "default": "auto",  # Options: "auto", "cuda", "cpu", "mps"
                })
            }
        }

    RETURN_TYPES = ("NARIMODEL", "STRING",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "Dia-TTS"

    def load(self, model_path, device):
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        device_torch = torch.device(device)
        print(f"Loading model from {model_path} on {device_torch}...")
        model = Dia.from_pretrained(model_path, device=device_torch)
        return (model,)

