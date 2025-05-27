import os
import re
import json
import base64
import time
from typing import List, Dict, Any, Union, Optional

from config import global_state, TEMP_DIR

# Fonction pour nettoyer les fichiers temporaires
def cleanup_temp_files():
    """Nettoie les fichiers temporaires créés pendant le traitement."""
    for key, path in global_state["image_paths"].items():
        if os.path.exists(path) and path.startswith(TEMP_DIR):
            try:
                os.remove(path)
            except Exception as e:
                print(f"Erreur lors du nettoyage du fichier {path}: {e}")

# Fonction pour libérer la mémoire utilisée par les modèles
def free_memory():
    """Libère la mémoire utilisée par les modèles."""
    from config import MODE
    
    if MODE == "mlx":
        if "model" in global_state:
            global_state["model"] = None
            global_state["processor"] = None
            global_state["config"] = None
            import gc
            gc.collect()

# Fonction pour extraire un JSON valide d'un texte
def extract_json_from_text(text):
    """Extrait le premier objet JSON valide d'un texte."""
    # Recherche la première séquence qui ressemble à un objet JSON
    json_pattern = r'(\{.*?\})'
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    if not matches:
        return None
    
    # Essayez chaque match jusqu'à trouver un JSON valide
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    return None

# Fonction pour convertir une image en base64
def image_to_base64(image_path: str) -> str:
    """
    Convertit une image en chaîne base64.
    
    Args:
        image_path (str): Chemin vers l'image à convertir
        
    Returns:
        str: Image encodée en base64
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string
    except Exception as e:
        print(f"Erreur lors de la conversion de l'image en base64: {str(e)}")
        return None

# Fonction pour obtenir le timestamp actuel
def get_timestamp():
    """Retourne un timestamp formaté."""
    return time.strftime("%Y-%m-%d %H:%M:%S")