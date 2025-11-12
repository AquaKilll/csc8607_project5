"""
Utils génériques.

Fonctions attendues (signatures imposées) :
- set_seed(seed: int) -> None
- get_device(prefer: str | None = "auto") -> str
- count_parameters(model) -> int
- save_config_snapshot(config: dict, out_dir: str) -> None
"""

import torch
import numpy as np
import random
import yaml
import os
from pathlib import Path


def set_seed(seed: int) -> None:
    """Initialise les seeds (numpy/torch/python)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(prefer: str | None = "auto") -> str:
    """Retourne 'cpu' ou 'cuda' (ou choix basé sur 'auto')."""
    if prefer == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    elif prefer == "cuda" and not torch.cuda.is_available():
        print("CUDA non disponible, utilisation du CPU")
        return "cpu"
    return prefer

def count_parameters(model) -> int:
    """Retourne le nombre de paramètres entraînables du modèle."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_config_snapshot(config: dict, out_dir: str) -> None:
    """Sauvegarde une copie de la config (YAML) dans out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    config_path = os.path.join(out_dir, "config_snapshot.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Configuration sauvegardée dans {config_path}")

def load_config(config_path: str) -> dict:
    """Charge un fichier de configuration YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config