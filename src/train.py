"""
Entraînement principal (à implémenter par l'étudiant·e).

Doit exposer un main() exécutable via :
    python -m src.train --config configs/config.yaml [--seed 42]

Exigences minimales :
- lire la config YAML
- respecter les chemins 'runs/' et 'artifacts/' définis dans la config
- journaliser les scalars 'train/loss' et 'val/loss' (et au moins une métrique de classification si applicable)
- supporter le flag --overfit_small (si True, sur-apprendre sur un très petit échantillon)
"""

import argparse
import torch
from src.data_loading import get_dataloaders
from src.model import BiLSTMAttention
from src.utils import load_config
import torch.nn as nn

def test_initial_loss():
    # Charger la configuration
    config = load_config("configs/config.yaml")

    # Charger les DataLoaders
    train_loader, _, _, _ = get_dataloaders(config)
    batch = next(iter(train_loader))
    inputs, labels = batch

    # Charger le modèle
    model = BiLSTMAttention(
        vocab_size=10002,
        embedding_dim=300,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        bidirectional=True
    )

    # Effectuer un forward pass
    logits = model(inputs)

    # Calculer la loss initiale
    criterion = nn.BCEWithLogitsLoss()
    # Ajuster la forme des labels pour correspondre à celle des logits
    loss = criterion(logits, labels.unsqueeze(1))
    print("Loss initiale :", loss.item())

    # Effectuer un pas de rétropropagation
    loss.backward()
    total_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print("Norme totale des gradients :", total_norm)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overfit_small", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    # Ajoutez d'autres arguments si nécessaire (batch_size, lr, etc.)
    args = parser.parse_args()
    # À implémenter par l'étudiant·e :
    raise NotImplementedError("train.main doit être implémenté par l'étudiant·e.")

if __name__ == "__main__":
    test_initial_loss()
    main()