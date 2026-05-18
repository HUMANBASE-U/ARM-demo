import os
from typing import Any, Dict

import torch
import yaml


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_checkpoint(path: str, payload: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    torch.save(payload, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    # Project checkpoints are self-produced and trusted in this workflow.
    return torch.load(path, map_location=map_location, weights_only=False)
