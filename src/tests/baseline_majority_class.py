"""
Baseline : Prédiction de la classe majoritaire.
"""
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
os.chdir(project_root)

import torch
from src.data_loading import get_dataloaders
from src.utils import load_config
from sklearn.metrics import accuracy_score, f1_score


def compute_majority_class_baseline():
    """
    Calcule la baseline en prédisant toujours la classe majoritaire.
    """
    # Charger la configuration et les DataLoaders
    config = load_config('configs/config.yaml')
    _, val_loader, test_loader, meta = get_dataloaders(config)
    
    # Déterminer la classe majoritaire (train)
    majority_class = 1  # Classe POSITIVE (50.1% dans train)
    print(f"Classe majoritaire prédite : {majority_class} (POSITIVE)")
    
    # Fonction pour évaluer un DataLoader
    def evaluate_baseline(loader, split_name):
        all_labels = []
        all_preds = []
        
        for _, labels in loader:
            all_labels.extend(labels.tolist())
            all_preds.extend([majority_class] * len(labels))
        
        # Calcul des métriques
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        print(f"\n=== Résultats pour {split_name} ===")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"F1 macro : {f1:.4f}")
        return accuracy, f1
    
    # Évaluer sur validation et test
    val_accuracy, val_f1 = evaluate_baseline(val_loader, "Validation")
    test_accuracy, test_f1 = evaluate_baseline(test_loader, "Test")
    
    print("\n=== Résumé des résultats ===")
    print(f"Validation - Accuracy : {val_accuracy:.4f}, F1 macro : {val_f1:.4f}")
    print(f"Test - Accuracy : {test_accuracy:.4f}, F1 macro : {test_f1:.4f}")


if __name__ == "__main__":
    compute_majority_class_baseline()