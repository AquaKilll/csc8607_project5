"""
Mini grid search — à implémenter.

Doit exposer un main() exécutable via :
    python -m src.grid_search --config configs/config.yaml

Exigences minimales :
- lire la section 'hparams' de la config
- lancer plusieurs runs en variant les hyperparamètres
- journaliser les hparams et résultats de chaque run (ex: TensorBoard HParams ou équivalent)
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from src.data_loading import get_dataloaders
from src.model import build_model
import yaml
import itertools

def grid_search(config_path):
    # Charger la configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Charger les DataLoaders
    train_loader, val_loader, _, meta = get_dataloaders(config)

    # Hyperparamètres à tester
    hparams = config['hparams']
    lr_values = hparams['lr']  # Liste des valeurs de learning rate
    wd_values = hparams['weight_decay']  # Liste des valeurs de weight decay
    hidden_sizes = hparams['hidden_sizes']  # Liste des tailles cachées
    num_layers = hparams['num_layers']  # Liste des nombres de couches

    # Combinaisons des hyperparamètres
    combinations = list(itertools.product(lr_values, wd_values, hidden_sizes, num_layers))

    # Initialiser TensorBoard
    writer = SummaryWriter(log_dir="runs/grid_search")

    # Seed pour reproductibilité
    torch.manual_seed(config['train']['seed'])

    for i, (lr, wd, hidden_size, num_layer) in enumerate(combinations):
        # Nom explicite du run
        run_name = f"lr={lr}_wd={wd}_hidden={hidden_size}_layers={num_layer}"
        writer.add_hparams({
            'lr': lr,
            'weight_decay': wd,
            'hidden_size': hidden_size,
            'num_layers': num_layer
        }, {})

        # Construire le modèle
        model_config = config['model']
        model_config['hidden_sizes'] = hidden_size
        model_config['num_layers'] = num_layer
        model = build_model(config)
        model.train()

        # Définir la loss et l'optimiseur
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        # Entraînement rapide (ex: 3 époques)
        for epoch in range(3):
            for inputs, labels in train_loader:
                labels = labels.view(-1, 1)  # Redimensionner les labels
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Évaluer sur validation
            val_loss = 0.0
            val_correct = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    labels = labels.view(-1, 1)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_correct += ((outputs > 0.5) == labels).sum().item()

            val_loss /= len(val_loader)
            val_accuracy = val_correct / meta['val_size']

            # Log dans TensorBoard
            writer.add_scalar(f"{run_name}/val_loss", val_loss, epoch)
            writer.add_scalar(f"{run_name}/val_accuracy", val_accuracy, epoch)

    writer.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    grid_search(args.config)

if __name__ == "__main__":
    main()