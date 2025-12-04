"""
Chargement des données.

Signature imposée :
get_dataloaders(config: dict) -> (train_loader, val_loader, test_loader, meta: dict)

Le dictionnaire meta doit contenir au minimum :
- "num_classes": int
- "input_shape": tuple (ex: (3, 32, 32) pour des images)
"""

"""
Chargement des données IMDb avec HuggingFace Datasets.
"""
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from datasets import load_dataset
from collections import Counter
import re


class IMDbDataset(Dataset):
    """Dataset wrapper pour IMDb."""
    
    def __init__(self, data, vocab, max_length):
        self.data = data
        self.vocab = vocab
        self.max_length = max_length
        
    def tokenize(self, text):
        """Tokenization simple."""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = item['label']  # 0 ou 1
        
        # Tokenization et conversion en indices
        tokens = self.tokenize(text)
        indices = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
        
        # Padding/Truncation
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        else:
            indices = indices + [self.vocab['<pad>']] * (self.max_length - len(indices))
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.float32)


def build_vocab(dataset, max_vocab_size=10000):
    """Construit le vocabulaire à partir du dataset."""
    counter = Counter()
    
    for item in dataset:
        text = item['text'].lower()
        tokens = re.findall(r'\b\w+\b', text)
        counter.update(tokens)
    
    # Les mots les plus fréquents
    most_common = counter.most_common(max_vocab_size)
    
    # Création du vocabulaire
    vocab = {'<pad>': 0, '<unk>': 1}
    for idx, (word, _) in enumerate(most_common, start=2):
        vocab[word] = idx
    
    return vocab


def get_dataloaders(config: dict):
    """
    Crée et retourne les DataLoaders d'entraînement/validation/test et des métadonnées.
    """
    dataset_config = config['dataset']
    train_config = config['train']
    
    # Paramètres
    max_length = dataset_config.get('max_length', 256)
    vocab_size = dataset_config.get('vocab_size', 10000)
    batch_size = train_config['batch_size']
    num_workers = dataset_config.get('num_workers', 0)  # 0 pour Windows
    seed = train_config['seed']
    
    print("Chargement du dataset IMDb depuis HuggingFace...")
    
    # Chargement avec HuggingFace Datasets
    dataset = load_dataset('imdb')
    
    train_data = dataset['train']
    test_data = dataset['test']
    
    print(f"Train original: {len(train_data)} exemples")
    print(f"Test: {len(test_data)} exemples")
    
    # Construction du vocabulaire
    print("Construction du vocabulaire...")
    vocab = build_vocab(train_data, max_vocab_size=vocab_size)
    print(f"Taille du vocabulaire: {len(vocab)}")
    
    # Split train/val
    train_size = int(dataset_config['split']['train'] * len(train_data))
    val_size = len(train_data) - train_size
    
    # Conversion en liste pour random_split
    train_list = list(train_data)
    
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        train_list, 
        [train_size, val_size],
        generator=generator
    )
    
    # Création des datasets
    train_dataset = IMDbDataset(train_subset, vocab, max_length)
    val_dataset = IMDbDataset(val_subset, vocab, max_length)
    test_dataset = IMDbDataset(test_data, vocab, max_length)
    
    # Création des DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    # Métadonnées
    meta = {
        "num_classes": 1,
        "input_shape": (max_length,),
        "vocab_size": len(vocab),
        "vocab": vocab,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "max_length": max_length,
        "class_names": ["negative", "positive"],
        "balanced": True
    }
    
    print(f"Train: {meta['train_size']} exemples")
    print(f"Val: {meta['val_size']} exemples")
    print(f"Test: {meta['test_size']} exemples")
    
    return train_loader, val_loader, test_loader, meta