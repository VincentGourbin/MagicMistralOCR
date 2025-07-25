"""
Configuration globale - Magic Document Scanner

Ce module contient la configuration globale de l'application, incluant :
- Variables d'état global
- Configuration des modes d'exécution (MLX/API)
- Paramètres des modèles et APIs
- Détection automatique de l'environnement

Variables principales:
- global_state: État partagé de l'application
- MODE: Mode d'exécution détecté (mlx/api/mlx_fallback)
- MODEL_NAME: Nom du modèle utilisé
"""

import os
import tempfile
import platform
import sys

# Désactiver les avertissements de tokenizers liés au parallélisme
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configuration de l'environnement
TEMP_DIR = tempfile.gettempdir()

# Variable globale pour stocker les données
global_state = {
    "file_paths": [],          # Liste des chemins de tous les fichiers
    "all_sections": [],        # Sections détectées dans les documents
    "image_paths": {},         # Stocke les chemins des images par (doc_index, page)
    "model": None,
    "processor": None,
    "config": None,
    "initialization_failed": False, # Indique si l'initialisation a échoué
    "api_config": {            # Configuration de l'API externe
        "server": os.environ.get("api_server", "https://api.mistral.ai/v1/chat/completions"),
        "model": os.environ.get("api_model", "mistral-small-latest"),
        "api_key": os.environ.get("api_key", ""),
        "enabled": bool(os.environ.get("api_key", "")),
        "pool_size": int(os.environ.get("api_pool_size", "5"))  # Taille du pool de threads
    },
    "MODE": "",               # Mode d'exécution explicitement stocké dans global_state
    "MODEL_NAME": ""          # Nom du modèle explicitement stocké dans global_state
}

# Détecter le mode d'exécution
is_mac_mlx = sys.platform == "darwin" and platform.processor() == 'arm'
# Vérifier si c'est un serveur local
is_local_server = any(local_host in global_state["api_config"]["server"] for local_host in ["localhost", "127.0.0.1", "0.0.0.0"])

# Pour les serveurs locaux, pas besoin de clé API
use_api = global_state["api_config"]["enabled"] and \
          global_state["api_config"]["server"] and \
          (is_local_server or global_state["api_config"]["api_key"])

# Définir le mode: MLX ou API
if is_mac_mlx:
    global_state["MODE"] = "mlx"
    global_state["MODEL_NAME"] = "mlx-community/Mistral-Small-3.1-24B-Instruct-2503-8bit"
elif use_api:
    global_state["MODE"] = "api"
    global_state["MODEL_NAME"] = global_state["api_config"]["model"]
else:
    global_state["MODE"] = "mlx_fallback"  # Fallback à MLX même sur non-Mac
    try:
        import mlx.core as mx
        global_state["MODEL_NAME"] = "mlx-community/Mistral-Small-3.1-24B-Instruct-2503-8bit"
    except ImportError:
        global_state["MODE"] = "error"
        global_state["MODEL_NAME"] = "Non disponible"

# Pour la compatibilité avec le code existant
MODE = global_state["MODE"]
MODEL_NAME = global_state["MODEL_NAME"]

# Fonction pour mettre à jour le statut API global
def refresh_api_status():
    """Met à jour le statut de l'API en fonction de la configuration actuelle."""
    global MODE, MODEL_NAME
    
    # Vérifier si c'est un serveur local
    is_local_server = any(local_host in global_state["api_config"]["server"] for local_host in ["localhost", "127.0.0.1", "0.0.0.0"])
    
    # Calculer si l'API est utilisable
    # Pour les serveurs locaux, pas besoin de clé API
    use_api = global_state["api_config"]["enabled"] and \
              global_state["api_config"]["server"] and \
              (is_local_server or global_state["api_config"]["api_key"])
    
    # Mode précédent pour le log
    previous_mode = global_state["MODE"]
    
    # Ajuster le mode si nécessaire
    if use_api:
        # Seulement mettre à jour si ce n'est pas déjà en mode API
        if global_state["MODE"] != "api":
            global_state["MODE"] = "api"
            global_state["MODEL_NAME"] = global_state["api_config"]["model"]
            # Mettre à jour les variables globales pour la compatibilité
            MODE = global_state["MODE"]
            MODEL_NAME = global_state["MODEL_NAME"]
            print(f"*** MODE CHANGÉ: {previous_mode} -> {global_state['MODE']} (API activée: {use_api}) ***")
    else:
        # Seulement mettre à jour si c'est actuellement en mode API
        if global_state["MODE"] == "api":
            # Revenir au mode par défaut
            if is_mac_mlx:
                global_state["MODE"] = "mlx"
            else:
                global_state["MODE"] = "mlx_fallback"
            global_state["MODEL_NAME"] = "mlx-community/Mistral-Small-3.1-24B-Instruct-2503-8bit"
            # Mettre à jour les variables globales pour la compatibilité
            MODE = global_state["MODE"]
            MODEL_NAME = global_state["MODEL_NAME"]
            print(f"*** MODE CHANGÉ: {previous_mode} -> {global_state['MODE']} (API activée: {use_api}) ***")

def show_runtime_info() -> str:
    """
    Affiche les informations sur le mode d'exécution actuel.
    
    Returns:
        str: Description détaillée du mode d'exécution et de la configuration
    """
    # Utiliser DIRECTEMENT global_state pour éviter les incohérences
    if global_state["MODE"] == "mlx":
        return "Exécution avec MLX sur Mac Apple Silicon - Modèle: mlx-community/Mistral-Small-3.1-24B-Instruct-2503-8bit"
    elif global_state["MODE"] == "api":
        api_config = global_state["api_config"]
        server = api_config["server"]
        model = api_config["model"]
        enabled = "Activée" if api_config["enabled"] else "Désactivée"
        masked_key = api_config["api_key"][:4] + "..." if api_config["api_key"] and len(api_config["api_key"]) > 4 else "Non définie"
        
        return f"Exécution via API externe ({enabled}) - Serveur: {server} - Modèle: {model} - Clé API: {masked_key}"
    elif global_state["MODE"] == "mlx_fallback":
        return "Exécution avec MLX en mode fallback - Modèle: mlx-community/Mistral-Small-3.1-24B-Instruct-2503-8bit"
    else:
        return "ERREUR: Aucun mode d'exécution disponible. Consultez les logs pour plus de détails."

# Force le rafraîchissement initial du mode
refresh_api_status()