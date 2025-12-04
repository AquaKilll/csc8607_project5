"""
Visualisation d'exemples après preprocessing pour le rapport.
"""
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
os.chdir(project_root)

import torch
from src.data_loading import get_dataloaders
from src.utils import load_config


def create_text_examples(loader, meta, num_examples=3):
    """Crée des exemples textuels formatés pour le rapport."""
    
    idx_to_word = {idx: word for word, idx in meta['vocab'].items()}
    inputs, labels = next(iter(loader))
    
    os.makedirs('artifacts', exist_ok=True)
    output_path = 'artifacts/preprocessing_examples.txt'
    
    print("\n" + "="*80)
    print("EXEMPLES APRÈS PREPROCESSING - IMDb Dataset")
    print("="*80)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("EXEMPLES APRÈS PREPROCESSING - IMDb Dataset\n")
        f.write("="*80 + "\n\n")
        
        for idx in range(num_examples):
            sequence = inputs[idx].tolist()
            label = labels[idx].item()
            label_name = "POSITIF" if label == 1 else "NÉGATIF"
            
            tokens = []
            for token_idx in sequence:
                if token_idx == meta['vocab']['<pad>']:
                    break
                tokens.append(idx_to_word.get(token_idx, '<unk>'))
            
            text = " ".join(tokens)
            
            # Affichage console
            print(f"\n--- Exemple {idx+1} ---")
            print(f"Label: {label_name} (valeur: {label})")
            print(f"Longueur: {len(tokens)} tokens (padding: {256-len(tokens)})")
            print(f"Texte (100 premiers caractères): {text[:100]}...")
            
            # Écriture fichier
            f.write(f"--- Exemple {idx+1} ---\n")
            f.write(f"Label: {label_name} (valeur: {label})\n")
            f.write(f"Longueur: {len(tokens)} tokens (padding: {256-len(tokens)})\n")
            f.write(f"Shape: (256,) - dtype: int64\n")
            f.write(f"Texte complet:\n{text}\n")
            f.write("\n" + "-"*80 + "\n\n")
    
    print("\n" + "="*80)
    print(f"✅ Exemples textuels sauvegardés: {output_path}")
    print("="*80 + "\n")


if __name__ == "__main__":
    config = load_config('configs/config.yaml')
    train_loader, val_loader, test_loader, meta = get_dataloaders(config)
    create_text_examples(train_loader, meta, num_examples=3)