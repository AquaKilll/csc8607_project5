"""
Construction du modèle (à implémenter par l'étudiant·e).

Signature imposée :
build_model(config: dict) -> torch.nn.Module
"""

"""
Construction du modèle BiLSTM avec Attention pour IMDb.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """Mécanisme d'attention simple."""
    
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_output):
        """
        Args:
            lstm_output: (batch, seq_len, hidden_size)
        Returns:
            context: (batch, hidden_size)
            attention_weights: (batch, seq_len)
        """
        # Calcul des scores d'attention
        scores = self.attention(lstm_output).squeeze(-1)  # (batch, seq_len)
        
        # Softmax pour obtenir les poids
        attention_weights = F.softmax(scores, dim=1)  # (batch, seq_len)
        
        # Moyenne pondérée
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, seq_len)
            lstm_output  # (batch, seq_len, hidden_size)
        ).squeeze(1)  # (batch, hidden_size)
        
        return context, attention_weights


class BiLSTMAttention(nn.Module):
    """Modèle BiLSTM avec attention pour classification binaire."""
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, 
                 dropout=0.3, bidirectional=True):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention layer
        lstm_output_size = hidden_size * self.num_directions
        self.attention = AttentionLayer(lstm_output_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Tête de classification
        self.fc = nn.Linear(lstm_output_size, 1)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len) - indices de tokens
        Returns:
            logits: (batch, 1) - logits pour classification binaire
        """
        # Embedding
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        embedded = self.dropout(embedded)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # lstm_out: (batch, seq_len, hidden_size * num_directions)
        
        # Attention
        context, attention_weights = self.attention(lstm_out)
        # context: (batch, hidden_size * num_directions)
        
        # Dropout
        context = self.dropout(context)
        
        # Classification
        logits = self.fc(context)  # (batch, 1)
        
        return logits


def build_model(config: dict) -> nn.Module:
    """
    Construit et retourne le modèle BiLSTM avec Attention selon la configuration.

    Args:
        config (dict): Configuration complète du modèle (extrait de configs/config.yaml).

    Returns:
        nn.Module: Modèle BiLSTM avec Attention.
    """
    model_config = config['model']

    # Récupération des paramètres depuis la configuration
    vocab_size = config['dataset']['vocab_size'] + 2  # +2 pour <unk> et <pad>
    embedding_dim = model_config['embedding_dim']
    hidden_size = model_config['hidden_sizes']
    num_layers = model_config['num_layers']
    dropout = model_config['dropout']
    bidirectional = model_config['bidirectional']
    
    model = BiLSTMAttention(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional
    )
    
    return model