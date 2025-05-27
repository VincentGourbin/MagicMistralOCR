import os
import json
from typing import List, Dict, Any, Optional

from config import TEMP_DIR, global_state
from model_handler import generate_from_image, load_model
from utils import extract_json_from_text

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