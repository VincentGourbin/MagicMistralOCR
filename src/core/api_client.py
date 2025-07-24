"""
Client API - Magic Document Scanner

Ce module gère les appels aux APIs externes :
- Appels REST vers les APIs de modèles (Mistral, OpenAI, etc.)
- Configuration et validation des paramètres API
- Gestion des erreurs et timeouts
- Support des serveurs locaux (LM Studio, Ollama)

Functions:
- call_external_api(): Effectue un appel à l'API externe
- update_api_config(): Met à jour la configuration API
"""

import os
import json
import requests
import pdf2image
from typing import Dict, Any

from core.config import global_state, TEMP_DIR, refresh_api_status
from utils.utils import image_to_base64

# Fonction pour effectuer un appel API externe au format OpenAI/Mistral
def call_external_api(image_path: str, prompt_text: str, page_num: int = 0) -> str:
    """
    Effectue un appel à l'API externe dans le format OpenAI/Mistral.
    
    Args:
        image_path (str): Chemin vers l'image à analyser
        prompt_text (str): Texte du prompt à envoyer à l'API
        page_num (int, optional): Numéro de page pour les PDF multi-pages
        
    Returns:
        str: Réponse de l'API (texte ou JSON)
    """
    try:
        
        api_config = global_state["api_config"]
        
        # Vérifier si c'est un serveur local
        is_local_server = any(local_host in api_config["server"] for local_host in ["localhost", "127.0.0.1", "0.0.0.0"])
        
        # Pour les serveurs locaux, on n'exige pas de clé API
        if not api_config["enabled"] or not api_config["server"]:
            return json.dumps({"error": "Configuration API manquante ou désactivée"})
        
        # Pour les serveurs externes, on exige une clé API
        if not is_local_server and not api_config["api_key"]:
            return json.dumps({"error": "Clé API requise pour les serveurs externes"})
        
        # Vérifier si le fichier est un PDF
        temp_img_path = None
        _, file_extension = os.path.splitext(image_path)
        if file_extension.lower() == '.pdf':
            # Convertir la page spécifiée du PDF en image
            images = pdf2image.convert_from_path(image_path, dpi=300)
            if not images or page_num >= len(images):
                return json.dumps({"error": "Impossible de convertir la page du PDF en image"})
            
            # Sauvegarder temporairement l'image de la page demandée
            temp_img_path = os.path.join(TEMP_DIR, f"temp_api_pdf_page_{page_num}.png")
            images[page_num].save(temp_img_path, "PNG", quality=100, dpi=(300, 300))
            image_path = temp_img_path
        
        # Convertir l'image en base64
        base64_image = image_to_base64(image_path)
        if not base64_image:
            return json.dumps({"error": "Échec de la conversion de l'image en base64"})
        
        # Créer la structure du message pour l'API
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        
        # Préparer la requête
        headers = {
            "Content-Type": "application/json"
        }
        
        # Ajouter l'authentification seulement si la clé API est fournie
        if api_config["api_key"]:
            headers["Authorization"] = f"Bearer {api_config['api_key']}"
        
        data = {
            "model": api_config["model"],
            "messages": messages,
            "max_tokens": 16384
        }
        
        
        # Envoi de la requête
        try:
            response = requests.post(
                api_config["server"],
                headers=headers,
                json=data,
                timeout=60  # Timeout de 60 secondes
            )
        except Exception as req_error:
            print(f"Erreur lors de la requête: {req_error}")
            return json.dumps({"error": f"Erreur de connexion: {str(req_error)}"})
        
        # Traitement de la réponse
        if response.status_code == 200:
            try:
                response_json = response.json()
                # Format de réponse OpenAI: extraire le texte de la première réponse
                if "choices" in response_json and len(response_json["choices"]) > 0:
                    message = response_json["choices"][0]["message"]
                    if "content" in message:
                        return message["content"]
                # Fallback au cas où la structure ne serait pas standard
                return json.dumps(response_json)
            except:
                return response.text
        else:
            error_msg = f"Erreur API (code {response.status_code}): {response.text}"
            if response.status_code == 404:
                print(f"🔍 DEBUG 404 - URL utilisée: {api_config['server']}")
                print(f"🔍 DEBUG 404 - Headers: {headers}")
                error_msg += f" [URL: {api_config['server']}]"
            return json.dumps({
                "error": error_msg
            })
            
    except Exception as e:
        print(f"Erreur lors de l'appel API: {str(e)}")
        return json.dumps({"error": f"Erreur lors de l'appel API: {str(e)}"})
    finally:
        # Nettoyer le fichier temporaire si créé
        if temp_img_path and os.path.exists(temp_img_path):
            try:
                os.remove(temp_img_path)
            except Exception as e:
                print(f"Erreur lors du nettoyage du fichier temporaire: {str(e)}")

# Fonction pour mettre à jour la configuration de l'API
def update_api_config(server: str, model: str, api_key: str, enabled: bool, pool_size: int = 5) -> str:
    """
    Met à jour la configuration de l'API externe.
    
    Args:
        server (str): URL du serveur API
        model (str): Nom du modèle à utiliser
        api_key (str): Clé API pour l'authentification
        enabled (bool): Activation/désactivation de l'API
        
    Returns:
        str: Message de statut après mise à jour
    """
    try:
        print("\n\n*** MISE À JOUR DE LA CONFIGURATION API ***")
        
        # Nettoyer les inputs
        clean_server = server.strip()
        clean_model = model.strip()
        clean_api_key = api_key.strip()
        
        # Vérifier si c'est un serveur local
        is_local_server = any(local_host in clean_server for local_host in ["localhost", "127.0.0.1", "0.0.0.0"])
        
        # Valider si l'API est réellement utilisable
        # Pour les serveurs locaux, pas besoin de clé API
        api_usable = enabled and clean_server and (is_local_server or clean_api_key)
        
        previous_mode = global_state["MODE"]
        print(f"État initial - Mode: {previous_mode}, API utilisable: {api_usable}")
        
        # Validation de la taille du pool
        pool_size = max(1, min(20, int(pool_size)))  # Limiter entre 1 et 20
        
        # Mise à jour de la configuration
        global_state["api_config"] = {
            "server": clean_server,
            "model": clean_model,
            "api_key": clean_api_key,
            "enabled": enabled,
            "pool_size": pool_size
        }
        
        # Modification DIRECTE du mode
        if api_usable:
            global_state["MODE"] = "api"
            global_state["MODEL_NAME"] = clean_model
        else:
            # Revenir au mode MLX si désactivé
            from core.config import is_mac_mlx
            if is_mac_mlx:
                global_state["MODE"] = "mlx"
            else:
                global_state["MODE"] = "mlx_fallback"
            global_state["MODEL_NAME"] = "mlx-community/Mistral-Small-3.1-24B-Instruct-2503-8bit"
        
        print(f"*** CHANGEMENT DE MODE DIRECT: {previous_mode} -> {global_state['MODE']} ***")
        
        # Message de confirmation
        if global_state["MODE"] == "api":
            return "API activée avec succès."
        else:
            return "API désactivée. Utilisation du modèle local MLX."
    except Exception as e:
        print(f"Erreur lors de la mise à jour de la configuration API: {str(e)}")
        return f"Erreur: {str(e)}"