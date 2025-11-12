"""
Data augmentation

Signature imposée :
get_augmentation_transforms(config: dict) -> objet/transform callable (ou None)
"""

def get_augmentation_transforms(config: dict):
    """Retourne les transformations d'augmentation. À implémenter."""
    # Pas d'augmentation pour ce projet (optionnel: synonym replacement, etc.)
    return None