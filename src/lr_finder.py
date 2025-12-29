"""
Recherche de taux d'apprentissage (LR finder) — à implémenter.

Doit exposer un main() exécutable via :
    python -m src.lr_finder --config configs/config.yaml

Exigences minimales :
- produire un log/trace permettant de visualiser (lr, loss) dans TensorBoard ou équivalent.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from src.data_loading import get_dataloaders
from src.model import build_model
import numpy as np
import yaml

def lr_finder(config_path):
    # Charger la configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Charger les DataLoaders
    train_loader, _, _, meta = get_dataloaders(config)

    # Initialiser le modèle
    model = build_model(config)
    model.train()

    # Définir la loss et l'optimiseur
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-7)  # LR initial très bas

    # Initialiser TensorBoard
    writer = SummaryWriter(log_dir="runs/lr_finder")

    # Variables pour le balayage LR
    lrs = []
    losses = []
    num_iterations = len(train_loader)
    lr_multiplier = np.exp(np.log(1e-1 / 1e-7) / num_iterations)  # Échelle logarithmique

    for i, (inputs, labels) in enumerate(train_loader):
        # Redimensionner les labels pour correspondre à la sortie du modèle
        labels = labels.view(-1, 1)  # Assure que labels a la forme (batch_size, 1)

        # Mettre à jour le LR
        lr = optimizer.param_groups[0]["lr"] * lr_multiplier
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Forward et backward
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Log dans TensorBoard
        writer.add_scalar("lr_finder/lr", lr, i)
        writer.add_scalar("lr_finder/loss", loss.item(), i)

        # Stocker les valeurs
        lrs.append(lr)
        losses.append(loss.item())

    writer.close()

    return lrs, losses

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    lr_finder(args.config)

if __name__ == "__main__":
    main()