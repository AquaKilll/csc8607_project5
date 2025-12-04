"""
Script pour analyser la distribution des classes du dataset IMDb.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import torch
import matplotlib.pyplot as plt
import numpy as np
from src.data_loading import get_dataloaders
from src.utils import load_config

def analyze_class_distribution():
    """Analyse et visualise la distribution des classes."""
    config = load_config('configs/config.yaml')
    train_loader, val_loader, test_loader, meta = get_dataloaders(config)
    
    # Compter les classes dans chaque split
    def count_classes(dataloader):
        pos_count = 0
        neg_count = 0
        for _, labels in dataloader:
            pos_count += (labels == 1).sum().item()
            neg_count += (labels == 0).sum().item()
        return neg_count, pos_count
    
    train_neg, train_pos = count_classes(train_loader)
    val_neg, val_pos = count_classes(val_loader)
    test_neg, test_pos = count_classes(test_loader)
    
    # Création du graphique
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    splits = ['Train', 'Validation', 'Test']
    data = [
        [train_neg, train_pos],
        [val_neg, val_pos],
        [test_neg, test_pos]
    ]
    
    for ax, split_name, counts in zip(axes, splits, data):
        bars = ax.bar(['Négatif', 'Positif'], counts, color=['#e74c3c', '#2ecc71'])
        ax.set_title(f'{split_name} ({sum(counts)} exemples)')
        ax.set_ylabel('Nombre d\'exemples')
        ax.set_ylim(0, max(max(counts) * 1.2, 15000))
        
        # Ajouter les pourcentages
        for bar, count in zip(bars, counts):
            percentage = (count / sum(counts)) * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                   f'{count}\n({percentage:.1f}%)', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('src/tests/class_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Tableau récapitulatif
    print("\n=== Distribution des classes ===")
    print(f"{'Split':<12} {'Négatif':<10} {'Positif':<10} {'Total':<10} {'Balance':<10}")
    print("-" * 60)
    for split_name, (neg, pos) in zip(splits, data):
        total = neg + pos
        balance = f"{(pos/total)*100:.1f}%"
        print(f"{split_name:<12} {neg:<10} {pos:<10} {total:<10} {balance:<10}")

if __name__ == "__main__":
    analyze_class_distribution()