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
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

# Définir le device (CPU ou GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def train_overfit_small(train_loader, model, criterion, optimizer, num_epochs=10):
    """Entraîne le modèle sur un petit sous-ensemble pour vérifier l'overfit."""
    writer = SummaryWriter(log_dir="runs/overfit_small")
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}")
        writer.add_scalar("train/loss", epoch_loss, epoch)
    writer.close()

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
    config = load_config(args.config)

    # Charger les DataLoaders et extraire le dataset d'entraînement
    train_loader, _, _, meta = get_dataloaders(config)
    train_dataset = train_loader.dataset  # Extraction du dataset d'entraînement

    if args.overfit_small:
        # Réduire le DataLoader à un petit sous-ensemble
        train_subset = torch.utils.data.Subset(train_dataset, range(64))  # 64 exemples
        train_loader = DataLoader(
            train_subset,
            batch_size=config['train']['batch_size'],
            shuffle=True,
            num_workers=config['dataset'].get('num_workers', 0),
            pin_memory=torch.cuda.is_available()
        )

        # Charger le modèle, la loss et l'optimiseur
        model = BiLSTMAttention(
            vocab_size=10002,
            embedding_dim=300,
            hidden_size=128,
            num_layers=2,
            dropout=0.3,
            bidirectional=True
        ).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

        # Entraîner le modèle
        train_overfit_small(train_loader, model, criterion, optimizer, num_epochs=10)
    else:
        raise NotImplementedError("L'entraînement sur l'ensemble complet doit être implémenté par l'étudiant·e.")

if __name__ == "__main__":
    test_initial_loss()
    main()