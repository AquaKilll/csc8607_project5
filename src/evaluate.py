"""
Évaluation — à implémenter.

Doit exposer un main() exécutable via :
    python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt

Exigences minimales :
- charger le modèle et le checkpoint
- calculer et afficher/consigner les métriques de test
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import torch
from src.data_loading import get_dataloaders
from src.model import BiLSTMAttention
from src.utils import load_config
from sklearn.metrics import accuracy_score, f1_score

def evaluate(config_path, checkpoint_path):
    # Charger la configuration
    config = load_config(config_path)

    # Charger le DataLoader de test
    _, _, test_loader, meta = get_dataloaders(config)

    # Charger le modèle
    model = BiLSTMAttention(
        vocab_size=10002,
        embedding_dim=100,
        hidden_size=config['model']['hidden_sizes'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        bidirectional=config['model']['bidirectional']
    ).to("cuda")
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    # Évaluation
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to("cuda"), labels.to("cuda")
            logits = model(inputs)
            predictions = (torch.sigmoid(logits) > 0.5).float()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    # Calcul des métriques
    accuracy = accuracy_score(all_labels, all_predictions)
    f1_macro = f1_score(all_labels, all_predictions, average="macro")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    evaluate(args.config, args.checkpoint)

if __name__ == "__main__":
    main()