import os
# Désactiver les avertissements de tokenizers liés au parallélisme
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gradio as gr
import sys
import tempfile
import json
import re
import platform
from PIL import Image
import pdf2image
import time as import_time
from io import BytesIO
import requests
import base64
from typing import List, Dict, Any, Union, Optional

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
        "enabled": bool(os.environ.get("api_key", ""))
    }
}

# Fonction pour mettre à jour le statut API global
def refresh_api_status():
    """Met à jour le statut de l'API en fonction de la configuration actuelle."""
    global use_api, MODE, MODEL_NAME
    
    use_api = global_state["api_config"]["enabled"] and \
              global_state["api_config"]["api_key"] and \
              global_state["api_config"]["server"]
    
    # Ajuster le mode si nécessaire
    if use_api and MODE != "api":
        MODE = "api"
        MODEL_NAME = global_state["api_config"]["model"]
    elif not use_api and MODE == "api":
        # Revenir au mode par défaut
        if is_mac_mlx:
            MODE = "mlx"
        else:
            MODE = "mlx_fallback"
        MODEL_NAME = "mlx-community/Mistral-Small-3.1-24B-Instruct-2503-8bit"

# Détecter le mode d'exécution
is_mac_mlx = sys.platform == "darwin" and platform.processor() == 'arm'
use_api = global_state["api_config"]["enabled"] and \
          global_state["api_config"]["api_key"] and \
          global_state["api_config"]["server"]

# Définir le mode: MLX ou API
if is_mac_mlx:
    MODE = "mlx"
    # Version MLX pour Mac Apple Silicon
    import mlx.core as mx
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    MODEL_NAME = "mlx-community/Mistral-Small-3.1-24B-Instruct-2503-8bit"
elif use_api:
    MODE = "api"
    MODEL_NAME = global_state["api_config"]["model"]
else:
    MODE = "mlx_fallback"  # Fallback à MLX même sur non-Mac
    try:
        import mlx.core as mx
        from mlx_vlm import load, generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config
        MODEL_NAME = "mlx-community/Mistral-Small-3.1-24B-Instruct-2503-8bit"
    except ImportError:
        MODE = "error"
        MODEL_NAME = "Non disponible"

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

# Fonction pour effectuer un appel API externe au format OpenAI/Mistral
def call_external_api(image_path: str, prompt_text: str) -> str:
    """
    Effectue un appel à l'API externe dans le format OpenAI/Mistral.
    
    Args:
        image_path (str): Chemin vers l'image à analyser
        prompt_text (str): Texte du prompt à envoyer à l'API
        
    Returns:
        str: Réponse de l'API (texte ou JSON)
    """
    try:
        api_config = global_state["api_config"]
        
        if not api_config["enabled"] or not api_config["api_key"] or not api_config["server"]:
            return json.dumps({"error": "Configuration API manquante ou désactivée"})
        
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
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    }
                ]
            }
        ]
        
        # Préparer la requête
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_config['api_key']}"
        }
        
        data = {
            "model": api_config["model"],
            "messages": messages,
            "max_tokens": 2048
        }
        
        # Envoi de la requête
        response = requests.post(
            api_config["server"],
            headers=headers,
            json=data,
            timeout=60  # Timeout de 60 secondes
        )
        
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
            return json.dumps({
                "error": f"Erreur API (code {response.status_code}): {response.text}"
            })
            
    except Exception as e:
        print(f"Erreur lors de l'appel API: {str(e)}")
        return json.dumps({"error": f"Erreur lors de l'appel API: {str(e)}"})

# Fonction pour mettre à jour la configuration de l'API
def update_api_config(server: str, model: str, api_key: str, enabled: bool) -> str:
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
        global MODE, MODEL_NAME, use_api
        
        # Mise à jour de la configuration
        global_state["api_config"] = {
            "server": server.strip(),
            "model": model.strip(),
            "api_key": api_key.strip(),
            "enabled": enabled
        }
        
        # Mise à jour explicite de use_api
        use_api = enabled and api_key.strip() and server.strip()
        
        # Rafraîchir le statut global
        refresh_api_status()
        
        return show_runtime_info()
    except Exception as e:
        print(f"Erreur lors de la mise à jour de la configuration API: {str(e)}")
        return f"Erreur: {str(e)}"

# Fonction pour charger le modèle selon l'environnement
def load_model():
    """Charge le modèle approprié selon le mode d'exécution actuel."""
    try:
        if global_state.get("initialization_failed", False):
            return None, None, None

        if MODE == "mlx":
            if global_state["model"] is None:
                model, processor = load(MODEL_NAME)
                config = load_config(MODEL_NAME)
                global_state["model"] = model
                global_state["processor"] = processor
                global_state["config"] = config
            return global_state["model"], global_state["processor"], global_state["config"]
        elif MODE == "api":
            # En mode API, nous n'avons pas besoin de charger un modèle
            return None, None, None
        else:  # MODE == "mlx_fallback" ou autre
            if global_state["model"] is None and MODE != "error":
                try:
                    model, processor = load(MODEL_NAME)
                    config = load_config(MODEL_NAME)
                    global_state["model"] = model
                    global_state["processor"] = processor
                    global_state["config"] = config
                except Exception as e:
                    print(f"Erreur lors du chargement du modèle MLX en fallback: {str(e)}")
                    global_state["initialization_failed"] = True
                    raise e
            return global_state["model"], global_state["processor"], global_state["config"]

    except Exception as e:
        print(f"Erreur critique lors du chargement du modèle: {str(e)}")
        global_state["initialization_failed"] = True
        global_state["model"] = None
        global_state["processor"] = None
        global_state["config"] = None
        raise e

# Fonction pour générer du texte à partir d'une image selon l'environnement
def generate_from_image(image_path: str, prompt_text: str, model=None, processor=None, config=None) -> str:
    """
    Génère du texte à partir d'une image en utilisant le modèle ou l'API approprié.
    
    Args:
        image_path (str): Chemin vers l'image à analyser
        prompt_text (str): Texte du prompt à envoyer au modèle
        model: Modèle MLX (si applicable)
        processor: Processeur MLX (si applicable)
        config: Configuration MLX (si applicable)
        
    Returns:
        str: Texte généré à partir de l'image
    """
    try:
        # Si l'initialisation a échoué, retourner une erreur
        if global_state.get("initialization_failed", False) and MODE != "api":
            return """{"error": "Le modèle n'a pas pu être chargé. Veuillez vérifier les logs."}"""
        
        if MODE == "api":
            # Utiliser l'API externe
            return call_external_api(image_path, prompt_text)
        else:
            # Version MLX
            formatted_prompt = apply_chat_template(
                processor, config, prompt_text, num_images=1
            )
            result = generate(model, processor, formatted_prompt, [image_path], verbose=False, max_tokens=2048, temperature=0.1)
            return result
            
    except Exception as e:
        print(f"Erreur lors de la génération d'image: {str(e)}")
        return f"""{"error": "Erreur: {str(e)}"}"""

# Fonction pour extraire les titres de section directement à partir de l'image
def extract_sections_from_image(image_path: str, model=None, processor=None, config=None) -> List[Dict[str, Any]]:
    """
    Extrait les titres de section à partir d'une image de document.
    
    Args:
        image_path (str): Chemin vers l'image à analyser
        model: Modèle MLX (si applicable)
        processor: Processeur MLX (si applicable)
        config: Configuration MLX (si applicable)
        
    Returns:
        List[Dict[str, Any]]: Liste des sections détectées avec leurs propriétés
    """
    try:
        # Préparation du prompt pour le modèle VLM avec demande explicite de format JSON
        prompt = """
        Examine cette image de document et extrait tous les titres de sections, champs ou entités présentes.
        
        Retourne UNIQUEMENT une liste au format JSON sous cette forme précise sans renvoyer les entités (field):
        {
          "sections": [
            {
              "title": "Nom du titre ou champ 1",
              "level": 1,
              "type": "section|header"
            },
            {
              "title": "Nom du titre ou champ 2",
              "level": 2,
              "type": "section|header"
            }
          ]
        }
        
        Assure-toi que le JSON est parfaitement formaté et inclut absolument TOUTES les sections ou champs visibles.
        Ne renvoie aucune explication, juste le JSON.
        """
        
        # Générer le résultat
        result = generate_from_image(image_path, prompt, model, processor, config)
        
        # Extraction du JSON à partir du résultat
        try:
            # Nettoyer le résultat pour s'assurer qu'il ne contient que du JSON valide
            json_str = result.strip()
            # Si le résultat contient des délimiteurs de code, les supprimer
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()
            
            # Parfois, le modèle peut ajouter du texte avant ou après le JSON
            # Essayer de trouver le début et la fin du JSON
            if "{" in json_str and "}" in json_str:
                start = json_str.find("{")
                end = json_str.rfind("}") + 1
                json_str = json_str[start:end]
                
            # Analyser le JSON
            data = json.loads(json_str)
            return data.get("sections", [])
        except Exception as e:
            print(f"Erreur lors de l'analyse du JSON: {e}")
            print(f"Résultat brut du modèle: {result}")
            # En cas d'échec, essayer d'extraire les lignes comme fallback
            lines = [line.strip() for line in result.split('\n') if line.strip() and not line.startswith('{') and not line.startswith('}')]
            sections = [{"title": line, "level": 1, "type": "section"} for line in lines]
            return sections
    except Exception as e:
        print(f"Erreur lors de l'extraction des sections: {str(e)}")
        return []

# Fonction pour extraire les valeurs des sections sélectionnées
def extract_section_values(image_path: str, selected_sections: List[str], model=None, processor=None, config=None, expert_prompt: str = "") -> List[Dict[str, Any]]:
    """
    Extrait les valeurs correspondant aux sections sélectionnées dans une image.
    
    Args:
        image_path (str): Chemin vers l'image à analyser
        selected_sections (List[str]): Liste des noms de sections à extraire
        model: Modèle MLX (si applicable)
        processor: Processeur MLX (si applicable)
        config: Configuration MLX (si applicable)
        expert_prompt (str, optional): Instructions supplémentaires pour l'extraction
        
    Returns:
        List[Dict[str, Any]]: Liste des valeurs extraites avec leur confiance
    """
    try:
        # Validation des entrées
        if not image_path or not os.path.exists(image_path):
            return []
            
        if not selected_sections or not isinstance(selected_sections, list):
            return []
            
        # Transformer les sections sélectionnées en texte pour le prompt
        sections_text = "\n".join([f"- {section}" for section in selected_sections])
        
        # Préparation du prompt pour extraire les valeurs des sections
        base_prompt = f"""
        Examine cette image de document et extrait les valeurs correspondant exactement aux champs ou sections suivants:
        
        {sections_text}
        
        Pour chaque section ou champ, trouve la valeur ou le contenu correspondant.
        Lorsque le contenu est une liste de valeur renvoi moi un tableau JSON avec les valeurs.
        Retourne UNIQUEMENT un objet JSON au format:
        
        {{
          "extracted_values": [
            {{
              "section": "Nom du champ 1",
              "value": "Valeur extraite 1",
              "confidence": 0.95
            }},
            {{
              "section": "Nom du champ 2",
              "value": "Valeur extraite 2",
              "confidence": 0.85
            }},
            {{
              "section": "Nom du champ 2",
              "value": ["Valeur extraite 2", "Valeur 3"],
              "confidence": 0.85
            }}
          ]
        }}
        
        Si tu ne trouves pas de valeur pour un champ, indique une chaîne vide pour "value" et 0 pour "confidence".
        Assure-toi que le JSON est parfaitement formaté. Ne renvoie aucune explication, juste le JSON.
        """
        
        # Ajouter les instructions du mode expert si présentes
        if expert_prompt and expert_prompt.strip():
            prompt = f"{base_prompt}\n\nInstructions supplémentaires:\n{expert_prompt}"
        else:
            prompt = base_prompt
        
        # Générer le résultat
        result = generate_from_image(image_path, prompt, model, processor, config)
        
        # Extraction du JSON à partir du résultat
        try:
            # Nettoyer le résultat pour s'assurer qu'il ne contient que du JSON valide
            json_str = result.strip()
            # Si le résultat contient des délimiteurs de code, les supprimer
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()
            
            # Parfois, le modèle peut ajouter du texte avant ou après le JSON
            if "{" in json_str and "}" in json_str:
                start = json_str.find("{")
                end = json_str.rfind("}") + 1
                json_str = json_str[start:end]
                
            # Analyser le JSON
            data = json.loads(json_str)
            return data.get("extracted_values", [])
        except Exception as e:
            print(f"Erreur lors de l'analyse du JSON des valeurs: {e}")
            print(f"Résultat brut du modèle: {result}")
            # En cas d'échec, renvoyer un tableau vide
            return []
    except Exception as e:
        print(f"Erreur lors de l'extraction des valeurs: {str(e)}")
        return []

# Fonction d'analyse de document pour MCP
def analyze_document(image_url: str) -> Dict[str, Any]:
    """
    Analyse un document fourni par URL pour en extraire les sections.
    
    Args:
        image_url (str): URL ou chemin vers l'image ou PDF à analyser
        
    Returns:
        Dict[str, Any]: Sections détectées dans le document
    """
    try:
        # Télécharger l'image si c'est une URL
        if image_url.startswith(("http://", "https://")):
            response = requests.get(image_url)
            temp_path = os.path.join(TEMP_DIR, "temp_download.jpg")
            with open(temp_path, "wb") as f:
                f.write(response.content)
            file_path = temp_path
        else:
            file_path = image_url
            
        # Charger le modèle
        model, processor, config = load_model()
        
        # Traiter le fichier
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()
        
        all_sections = []
        
        if file_extension in ['.pdf']:
            # Convertir PDF en images
            images = pdf2image.convert_from_path(file_path)
            
            # Traiter jusqu'à 5 premières pages
            for i, img in enumerate(images[:5]):
                # Sauvegarder temporairement l'image
                temp_img_path = os.path.join(TEMP_DIR, f"temp_doc0_page_{i}.png")
                img.save(temp_img_path, "PNG", quality=95, dpi=(300, 300))
                
                # Traiter l'image avec le modèle
                page_sections = extract_sections_from_image(temp_img_path, model, processor, config)
                
                # Ajouter les titres trouvés avec indication de la page
                for section in page_sections:
                    section["page"] = i + 1
                    all_sections.append(section)
                
                # Nettoyer l'image temporaire
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
        else:
            # Pour les images directement
            all_sections = extract_sections_from_image(file_path, model, processor, config)
            for section in all_sections:
                section["page"] = 1
        
        # Nettoyer le fichier temporaire si c'était une URL
        if image_url.startswith(("http://", "https://")) and os.path.exists(temp_path):
            os.remove(temp_path)
        
        return {"sections": all_sections, "total": len(all_sections)}
    
    except Exception as e:
        print(f"Erreur lors de l'analyse du document: {str(e)}")
        return {"error": str(e), "sections": []}

# Fonction d'extraction de valeurs pour MCP
def extract_values(image_url: str, sections: List[str], expert_instructions: str = "") -> Dict[str, Any]:
    """
    Extrait les valeurs des sections spécifiées dans un document.
    
    Args:
        image_url (str): URL ou chemin vers l'image ou PDF à analyser
        sections (List[str]): Liste des sections dont les valeurs doivent être extraites
        expert_instructions (str, optional): Instructions supplémentaires pour l'extraction
        
    Returns:
        Dict[str, Any]: Valeurs extraites pour chaque section
    """
    try:
        # Télécharger l'image si c'est une URL
        if image_url.startswith(("http://", "https://")):
            response = requests.get(image_url)
            temp_path = os.path.join(TEMP_DIR, "temp_download.jpg")
            with open(temp_path, "wb") as f:
                f.write(response.content)
            file_path = temp_path
        else:
            file_path = image_url
            
        # Charger le modèle
        model, processor, config = load_model()
        
        # Extraire les valeurs
        expert_text = expert_instructions if expert_instructions else ""
        extracted_values = extract_section_values(file_path, sections, model, processor, config, expert_text)
        
        # Nettoyer le fichier temporaire si c'était une URL
        if image_url.startswith(("http://", "https://")) and os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Transformer la sortie en dictionnaire pour une meilleure lisibilité
        result = {}
        for item in extracted_values:
            result[item["section"]] = {
                "value": item["value"],
                "confidence": item["confidence"]
            }
        
        return {"extracted_values": extracted_values, "results": result}
    
    except Exception as e:
        print(f"Erreur lors de l'extraction des valeurs: {str(e)}")
        return {"error": str(e), "extracted_values": []}

# Fonction pour scanner et extraire les sections d'un document pour l'interface Gradio
def magic_scan(file):
    """
    Analyse un document pour détecter automatiquement les sections et titres qu'il contient.
    
    Args:
        file: Fichier à analyser (PDF ou image)
        
    Returns:
        tuple: Un tuple contenant (1) les sections détectées sous forme de liste pour le composant CheckboxGroup 
              et (2) le chemin vers le fichier JSON contenant les résultats détaillés
    """    
    try:
        if file is None:
            return gr.update(choices=[]), None
        
        # Vérifier le fichier
        file_path = file.name
        global_state["file_paths"] = [file_path]  # Pour le moment, un seul fichier
        _, file_extension = os.path.splitext(file_path)
        
        # Charger le modèle selon l'environnement
        model, processor, config = load_model()
        
        # Si l'initialisation a échoué et qu'aucun modèle n'est disponible (sauf en mode API)
        if global_state.get("initialization_failed", False) and MODE != "api":
            return gr.update(choices=["Erreur: Le modèle n'a pas pu être chargé. Voir les logs pour plus de détails."]), None
        
        # Réinitialiser les chemins d'image et nettoyer les anciens fichiers
        cleanup_temp_files()
        global_state["image_paths"] = {}
        
        # Traiter selon le type de fichier
        all_sections = []
        
        if file_extension.lower() == '.pdf':
            # Convertir PDF en images
            images = pdf2image.convert_from_path(file_path)
            
            # Traiter jusqu'à 5 premières pages pour plus de couverture
            for i, img in enumerate(images[:5]):
                # Sauvegarder temporairement l'image
                temp_img_path = os.path.join(TEMP_DIR, f"temp_doc0_page_{i}.png")
                img.save(temp_img_path, "PNG", quality=95, dpi=(300, 300))
                
                # Stocker le chemin de l'image pour utilisation ultérieure
                global_state["image_paths"][(0, i+1)] = temp_img_path
                
                # Traiter l'image avec le modèle
                page_sections = extract_sections_from_image(temp_img_path, model, processor, config)
                
                # Ajouter les titres trouvés avec indication de la page
                for section in page_sections:
                    section["page"] = i + 1
                    section["doc_index"] = 0
                    all_sections.append(section)
        else:
            # Pour les images directement
            global_state["image_paths"][(0, 1)] = file_path
            all_sections = extract_sections_from_image(file_path, model, processor, config)
            for section in all_sections:
                section["page"] = 1
                section["doc_index"] = 0
        
        # Dédupliquer les titres
        seen_titles = set()
        unique_sections = []
        for section in all_sections:
            if section["title"] not in seen_titles:
                seen_titles.add(section["title"])
                unique_sections.append(section)
        
        # Stocker les sections dans l'état global
        global_state["all_sections"] = unique_sections
        
        # Préparer les choix pour le composant CheckboxGroup
        section_choices = [
            f"{section['title']} (Niveau: {section.get('level', 1)}, Type: {section.get('type', 'section')}, Page: {section.get('page', 1)})"
            for section in unique_sections
        ]
        
        # Sauvegarder le JSON pour un traitement ultérieur
        json_result = {"sections": unique_sections}
        json_path = os.path.join(TEMP_DIR, "extracted_sections.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_result, f, ensure_ascii=False, indent=2)
        
        return gr.update(choices=section_choices, value=[]), json_path
    except Exception as e:
        print(f"Erreur lors du scan: {str(e)}")
        return gr.update(choices=[f"Erreur: {str(e)}"]), None

# Fonction pour ajouter des sections manuellement
def add_manual_sections(section_names, existing_sections):
    """
    Ajoute des sections définies manuellement à la liste des sections existantes.
    
    Args:
        section_names (str): Texte contenant les noms des sections à ajouter, une par ligne
        existing_sections: Composant Gradio contenant les sections déjà détectées ou ajoutées
        
    Returns:
        gr.update: Mise à jour du composant CheckboxGroup avec les nouvelles sections ajoutées
    """    
    try:
        # Validation des entrées
        if not isinstance(section_names, str) or not section_names.strip():
            return gr.update()
        
        # Récupérer les sections existantes
        current_choices = existing_sections.choices if hasattr(existing_sections, 'choices') else []
        current_values = existing_sections.value if hasattr(existing_sections, 'value') else []
        
        # Traiter les nouvelles sections (une par ligne)
        new_sections = []
        for section in section_names.split('\n'):
            section = section.strip()
            if section and len(section) <= 100:  # Limite raisonnable
                new_sections.append(section)
        
        # Ajouter les nouvelles sections
        new_choices = list(current_choices)
        for section in new_sections:
            formatted_section = f"{section} (Niveau: 1, Type: manual, Page: 1)"
            if formatted_section not in new_choices:
                new_choices.append(formatted_section)
                # Ajouter à l'état global
                global_state["all_sections"].append({
                    "title": section,
                    "level": 1,
                    "type": "manual",
                    "page": 1,
                    "doc_index": -1  # Indique une entrée manuelle
                })
        
        # Mettre à jour le composant CheckboxGroup
        return gr.update(choices=new_choices, value=current_values)
    except Exception as e:
        print(f"Erreur lors de l'ajout de sections manuelles: {str(e)}")
        return gr.update()

# Fonction pour traiter plusieurs documents via l'interface Gradio
def process_multiple_documents(files, selected_sections, expert_mode_enabled=False, expert_prompt=""):
    """
    Extrait les valeurs des sections sélectionnées à partir d'un ou plusieurs documents.
    
    Args:
        files (list): Liste des fichiers (PDF ou images) à traiter
        selected_sections (list): Liste des sections dont les valeurs doivent être extraites
        expert_mode_enabled (bool): Indique si le mode expert est activé pour personnaliser l'extraction
        expert_prompt (str): Instructions supplémentaires pour l'extraction en mode expert
        
    Returns:
        tuple: Un tuple contenant (1) un résumé textuel des extractions réalisées 
              et (2) le chemin vers le fichier JSON contenant les résultats détaillés
    """    
    try:
        if not files or not selected_sections:
            return "Veuillez sélectionner des documents et des sections à extraire.", None
        
        # Charger le modèle
        model, processor, config = load_model()
        
        # Si l'initialisation a échoué et qu'aucun modèle n'est disponible (sauf en mode API)
        if global_state.get("initialization_failed", False) and MODE != "api":
            return "Erreur: Le modèle n'a pas pu être chargé. Voir les logs pour plus de détails.", None
        
        # Extraire les titres des sections sélectionnées
        section_titles = []
        for section in selected_sections:
            # Extraire le titre (tout ce qui est avant " (Niveau:")
            if " (Niveau:" in section:
                title = section.split(" (Niveau:")[0].strip()
            else:
                title = section.strip()
                
            # Enlever les guillemets des titres si présents
            if title.startswith('"') and title.endswith('"'):
                title = title[1:-1]
            elif title.startswith('"title": "') and title.endswith('"'):
                title = title[9:-1]
                
            section_titles.append(title)
        
        # Résultats pour tous les documents
        all_results = []
        
        # Traiter chaque fichier
        for doc_index, file in enumerate(files):
            file_path = file.name
            _, file_extension = os.path.splitext(file_path)
            
            # Résultat pour ce document
            doc_result = {
                "document": os.path.basename(file_path),
                "extracted_values": []
            }
            
            # Traiter selon le type de fichier
            if file_extension.lower() == '.pdf':
                # Convertir PDF en images
                images = pdf2image.convert_from_path(file_path)
                
                # Traiter jusqu'à 5 premières pages
                for page_num, img in enumerate(images[:5]):
                    # Sauvegarder temporairement l'image
                    temp_img_path = os.path.join(TEMP_DIR, f"temp_doc{doc_index}_page_{page_num}.png")
                    img.save(temp_img_path, "PNG", quality=95, dpi=(300, 300))
                    
                    # Extraire les valeurs des sections pour cette page
                    expert_text = expert_prompt if expert_mode_enabled else ""
                    page_values = extract_section_values(temp_img_path, section_titles, model, processor, config, expert_text)
                    
                    # Ajouter le numéro de page à chaque résultat
                    for value in page_values:
                        value["page"] = page_num + 1
                    
                    # Ajouter à la liste des valeurs extraites
                    doc_result["extracted_values"].extend(page_values)
                    
                    # Nettoyage
                    if os.path.exists(temp_img_path):
                        os.remove(temp_img_path)
            else:
                # Pour les images directement
                expert_text = expert_prompt if expert_mode_enabled else ""
                page_values = extract_section_values(file_path, section_titles, model, processor, config, expert_text)
                
                # Ajouter le numéro de page
                for value in page_values:
                    value["page"] = 1
                
                # Ajouter à la liste des valeurs extraites
                doc_result["extracted_values"].extend(page_values)
            
            # Ajouter les résultats de ce document
            all_results.append(doc_result)
        
        # Organiser les résultats
        result = {
            "timestamp": import_time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": MODE,  # Indiquer le mode utilisé
            "expert_mode": expert_mode_enabled,
            "documents": all_results
        }
        
        # Sauvegarder en JSON
        json_path = os.path.join(TEMP_DIR, "multi_doc_extracted_values.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # Préparer un résumé pour l'affichage
        summary = f"Extraction réussie pour {len(files)} documents (Mode: {MODE}"
        if expert_mode_enabled:
            summary += ", Mode Expert activé"
        summary += "):\n\n"
        
        for doc_result in all_results:
            doc_name = doc_result["document"]
            values = doc_result["extracted_values"]
            summary += f"Document: {doc_name} ({len(values)} valeurs extraites)\n"
            
            for item in values:
                confidence = item.get("confidence", 0) * 100
                page = item.get("page", 1)
                summary += f"  • {item['section']} (p.{page}): {item['value']} (confiance: {confidence:.0f}%)\n"
            
            summary += "\n"
        
        # Nettoyer les fichiers temporaires
        cleanup_temp_files()
        
        # Libérer la mémoire du modèle si possible
        free_memory()
        
        return summary, json_path
    except Exception as e:
        print(f"Erreur lors du traitement multiple: {str(e)}")
        # Nettoyer en cas d'erreur
        cleanup_temp_files()
        free_memory()
        return f"Erreur lors de l'extraction: {str(e)}", None

# Fonction pour afficher l'info sur le mode d'exécution actuel
def show_runtime_info():
    if MODE == "mlx":
        return "Exécution avec MLX sur Mac Apple Silicon - Modèle: mlx-community/Mistral-Small-3.1-24B-Instruct-2503-8bit"
    elif MODE == "api":
        api_config = global_state["api_config"]
        server = api_config["server"]
        model = api_config["model"]
        enabled = "Activée" if api_config["enabled"] else "Désactivée"
        masked_key = api_config["api_key"][:4] + "..." if api_config["api_key"] and len(api_config["api_key"]) > 4 else "Non définie"
        
        return f"Exécution via API externe ({enabled}) - Serveur: {server} - Modèle: {model} - Clé API: {masked_key}"
    elif MODE == "mlx_fallback":
        return "Exécution avec MLX en mode fallback - Modèle: mlx-community/Mistral-Small-3.1-24B-Instruct-2503-8bit"
    else:
        return "ERREUR: Aucun mode d'exécution disponible. Consultez les logs pour plus de détails."

# Interface Gradio
with gr.Blocks(title="Magic Document Scanner") as app:
    gr.Markdown("# 📄 Magic Document Scanner")
    runtime_info_text = gr.Markdown(show_runtime_info())
    
    # Interface de configuration de l'API
    with gr.Accordion("⚙️ Configuration API externe", open=False):
        with gr.Row():
            with gr.Column(scale=3):
                api_server = gr.Textbox(
                    label="URL du serveur API", 
                    placeholder="https://api.mistral.ai/v1/chat/completions",
                    value=global_state["api_config"]["server"]
                )
                api_model = gr.Textbox(
                    label="Modèle à utiliser", 
                    placeholder="mistral-small-latest",
                    value=global_state["api_config"]["model"]
                )
                api_key = gr.Textbox(
                    label="Clé API", 
                    placeholder="Votre clé API",
                    value=global_state["api_config"]["api_key"],
                    type="password"
                )
            with gr.Column(scale=1):
                api_enabled = gr.Checkbox(
                    label="Activer l'API externe", 
                    value=global_state["api_config"]["enabled"]
                )
                update_api_button = gr.Button("💾 Sauvegarder la configuration API", variant="primary")
    
    with gr.Tab("1️⃣ Configurer les sections"):
        with gr.Row():
            with gr.Column(scale=2):
                file_input = gr.File(label="Document modèle (PDF ou Image)")
                scan_button = gr.Button("🔍 Magic Scan", variant="primary")
            
            with gr.Column(scale=1):
                manual_sections = gr.Textbox(
                    label="Ajouter des sections manuellement (une par ligne)",
                    placeholder="Nom\nPrénom\nAdresse\nEmail\n...",
                    lines=6
                )
                add_sections_button = gr.Button("➕ Ajouter ces sections")
        
        with gr.Row():
            sections_output = gr.CheckboxGroup(
                label="Sections détectées (cochez celles à extraire)",
                choices=[],
                interactive=True
            )
            json_output = gr.File(label="Résultat JSON des sections")
    
    with gr.Tab("2️⃣ Extraire les valeurs"):
        with gr.Row():
            files_input = gr.File(
                label="Documents à traiter (PDF ou Images)",
                file_count="multiple"
            )
        
        with gr.Accordion("🔧 Mode Expert", open=False):
            expert_mode_enabled = gr.Checkbox(
                label="Activer le mode expert", 
                value=False
            )
            expert_prompt = gr.Textbox(
                label="Instructions supplémentaires (prompt personnalisé)",
                placeholder="Ajoutez ici des instructions spécifiques pour guider l'extraction, par exemple:\n- Considérer les valeurs à proximité immédiate du champ\n- Ignorer les valeurs barrées ou en italique\n- Rechercher aussi dans les tableaux\n- Considérer le texte en pied de page",
                lines=5
            )
        
        extract_button = gr.Button("✨ Extraire les valeurs des sections sélectionnées", variant="primary")
        extraction_results = gr.Textbox(label="Résultats de l'extraction", lines=15)
        values_json_output = gr.File(label="Résultat JSON des valeurs extraites")
    
    # Connecter les événements
    update_api_button.click(
        fn=update_api_config,
        inputs=[api_server, api_model, api_key, api_enabled],
        outputs=runtime_info_text
    )
    
    scan_button.click(
        fn=magic_scan,
        inputs=file_input,
        outputs=[sections_output, json_output]
    )
    
    add_sections_button.click(
        fn=add_manual_sections,
        inputs=[manual_sections, sections_output],
        outputs=sections_output
    )
    
    extract_button.click(
        fn=process_multiple_documents,
        inputs=[files_input, sections_output, expert_mode_enabled, expert_prompt],
        outputs=[extraction_results, values_json_output]
    )
    
    # Exposer les fonctions MCP
    with gr.Tab("🔌 API et MCP"):
        gr.Markdown("""
        ### API et MCP (Model Control Protocol)
        
        Magic Document Scanner expose des fonctionnalités via le protocole MCP, permettant à des assistants IA (comme Claude Desktop) d'utiliser directement les capacités d'extraction de documents.
        
        #### Outils disponibles via MCP
        
        1. **analyze_document** - Analyse un document pour en extraire les sections
        2. **extract_values** - Extrait les valeurs des sections spécifiées dans un document
        
        #### Comment utiliser
        
        1. Lancez Magic Document Scanner avec l'option MCP activée
        2. Ajoutez l'URL du serveur MCP dans votre client MCP (par ex. Claude Desktop)
        3. Demandez à votre assistant d'analyser des documents ou d'extraire des informations
        
        #### Exemple de configuration MCP
        
        ```json
        {
          "mcpServers": {
            "magic-scanner": {
              "url": "http://localhost:7860/gradio_api/mcp/sse"
            }
          }
        }
        ```
        """)
    
    gr.Markdown(f"""
    ### Mode d'emploi :
    1. **Configurer l'API externe (optionnel)** :
       - Cliquez sur "Configuration API externe" pour ouvrir le panneau
       - Entrez l'URL du serveur API (format: https://api.mistral.ai/v1/chat/completions)
       - Spécifiez le modèle à utiliser (ex: mistral-small-latest)
       - Saisissez votre clé API et activez l'option
       - Cliquez sur "Sauvegarder la configuration API"
       
    2. **Configurer les sections** : 
       - Téléchargez un document modèle et cliquez sur "Magic Scan" pour détecter automatiquement les sections.
       - OU ajoutez manuellement des sections en les saisissant (une par ligne) et en cliquant sur "Ajouter ces sections".
       - Cochez les sections que vous souhaitez extraire.
    
    3. **Extraire les valeurs** :
       - Téléchargez un ou plusieurs documents du même type.
       - Pour une extraction avancée, utilisez le mode expert pour personnaliser les instructions d'extraction.
       - Cliquez sur "Extraire les valeurs" pour obtenir les informations des sections sélectionnées.
       - Les résultats sont disponibles au format JSON.
       
    4. **Utiliser le serveur MCP (pour LLMs)** :
       - Les assistants comme Claude Desktop peuvent utiliser directement cette application
       - Ajoutez l'URL MCP dans la configuration de votre assistant IA
       
    ### Informations techniques :
    - **Mode d'exécution actuel**: {MODE}
    - **Modèle**: {MODEL_NAME}
    - Supporte les PDF multi-pages (traitement des 5 premières pages)
    - Compatible avec les API externes au format OpenAI (Mistral, OpenAI, etc.)
    - Expose un serveur MCP pour l'intégration avec les LLMs
    """)

# Exposer les fonctions MCP pour les assistants IA
analyze_document_for_mcp = analyze_document
extract_values_from_document = extract_values

# Lancer l'application
if __name__ == "__main__":
    app.launch(mcp_server=True)  # Activer le serveur MCP