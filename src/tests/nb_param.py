from src.model import BiLSTMAttention
from src.utils import count_parameters

# Initialisation du modèle avec les hyperparamètres
model = BiLSTMAttention(
    vocab_size=10002, 
    embedding_dim=300, 
    hidden_size=128, 
    num_layers=2, 
    dropout=0.3, 
    bidirectional=True
)

# Calcul du nombre de paramètres
print("Nombre total de paramètres :", count_parameters(model))