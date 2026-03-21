import torch
import os
from pathlib import Path
from datetime import datetime

def save_weights(model_name: str, model: torch.nn.Module, path: os.PathLike):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), f"{path}/{model_name}_{datetime.now().strftime('%Y-%m-%d_%Hh%m-%Ss')}.pth")

def load_weights(model: torch.nn.Module, path: os.PathLike):
    model.load_state_dict(torch.load(path, weights_only=True))