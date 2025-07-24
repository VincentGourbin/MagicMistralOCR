"""
Gestionnaire de modèles - Magic Document Scanner

Ce module gère le chargement et l'utilisation des modèles :
- Modèles MLX locaux (Apple Silicon)
- APIs externes (Mistral, OpenAI, etc.)
- Sélection automatique selon l'environnement

Functions:
- load_model(): Charge le modèle approprié selon le mode
- generate_from_image(): Génère du texte à partir d'une image
"""

import os
import json
from typing import Tuple, Any, Optional

from core.config import global_state, MODE, MODEL_NAME, TEMP_DIR

# Fonction pour charger le modèle selon l'environnement
def load_model():
    """Charge le modèle approprié selon le mode d'exécution actuel."""
    try:
        if global_state.get("initialization_failed", False):
            return None, None, None

        # Vérifier DIRECTEMENT le mode d'exécution dans global_state
        
        # Si on est en mode API, ne pas charger le modèle MLX
        if global_state["MODE"] == "api":
            print("Mode API actif: pas de chargement du modèle MLX")
            return None, None, None
        
        # Sinon, chargement du modèle MLX selon le mode normal
        if global_state["MODE"] == "mlx":
            if global_state["model"] is None:
                print("Chargement du modèle MLX...")
                from mlx_vlm import load, generate
                from mlx_vlm.utils import load_config
                
                model, processor = load(global_state["MODEL_NAME"])
                config = load_config(global_state["MODEL_NAME"])
                global_state["model"] = model
                global_state["processor"] = processor
                global_state["config"] = config
            return global_state["model"], global_state["processor"], global_state["config"]
        else:  # MODE == "mlx_fallback" ou autre
            if global_state["model"] is None and global_state["MODE"] != "error":
                try:
                    print("Chargement du modèle MLX en fallback...")
                    from mlx_vlm import load, generate
                    from mlx_vlm.utils import load_config
                    
                    model, processor = load(global_state["MODEL_NAME"])
                    config = load_config(global_state["MODEL_NAME"])
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
def generate_from_image(image_path: str, prompt_text: str, model=None, processor=None, config=None, page_num: int = 0) -> str:
    """
    Génère du texte à partir d'une image en utilisant le modèle ou l'API approprié.
    
    Args:
        image_path (str): Chemin vers l'image à analyser
        prompt_text (str): Texte du prompt à envoyer au modèle
        model: Modèle MLX (si applicable)
        processor: Processeur MLX (si applicable)
        config: Configuration MLX (si applicable)
        page_num (int, optional): Numéro de page pour les PDF multi-pages
        
    Returns:
        str: Texte généré à partir de l'image
    """
    try:
        # Si l'initialisation a échoué, retourner une erreur
        if global_state.get("initialization_failed", False) and global_state["MODE"] != "api":
            return json.dumps({"error": "Le modèle n'a pas pu être chargé. Veuillez vérifier les logs."})
        
        # Logging explicite du mode utilisé
        current_mode = global_state["MODE"]
        
        # Vérifier si le fichier est un PDF
        temp_img_path = None
        _, file_extension = os.path.splitext(image_path)
        if file_extension.lower() == '.pdf':
            # Convertir la page spécifiée du PDF en image
            import pdf2image
            images = pdf2image.convert_from_path(image_path, dpi=300)
            if not images or page_num >= len(images):
                return json.dumps({"error": "Impossible de convertir la page du PDF en image"})
            
            # Sauvegarder temporairement l'image de la page demandée
            temp_img_path = os.path.join(TEMP_DIR, f"temp_gen_pdf_page_{page_num}.png")
            images[page_num].save(temp_img_path, "PNG", quality=100, dpi=(300, 300))
            image_path = temp_img_path
        
        try:
            import time
            start_time = time.time()
            
            # Utiliser le mode stocké dans global_state, pas la variable globale
            if global_state["MODE"] == "api":
                # Utiliser l'API externe
                from core.api_client import call_external_api
                result = call_external_api(image_path, prompt_text, page_num)
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"⏱️  Appel API: {elapsed_time:.2f}s")
                
                return result
            else:
                # Version MLX
                from mlx_vlm.prompt_utils import apply_chat_template
                from mlx_vlm import generate
                
                formatted_prompt = apply_chat_template(
                    processor, config, prompt_text, num_images=1
                )
                result = generate(model, processor, formatted_prompt, [image_path], verbose=False, max_tokens=16384, temperature=0.1)
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                # Extraire seulement le texte (premier élément du tuple)
                if isinstance(result, tuple):
                    generated_text = result[0]
                    metadata = result[1]
                    
                    # Afficher les métriques avec le temps
                    tokens_used = metadata.get('total_tokens', 'N/A')
                    generation_speed = metadata.get('generation_tps', 'N/A')
                    print(f"⏱️  Génération MLX: {elapsed_time:.2f}s | {tokens_used} tokens | {generation_speed} t/s")
                    
                    return generated_text
                else:
                    # Au cas où le format changerait dans une future version
                    print(f"⏱️  Génération MLX: {elapsed_time:.2f}s")
                    return result
        finally:
            # Nettoyer le fichier temporaire si créé
            if temp_img_path and os.path.exists(temp_img_path):
                try:
                    os.remove(temp_img_path)
                except Exception as e:
                    print(f"Erreur lors du nettoyage du fichier temporaire: {str(e)}")
            
    except Exception as e:
        print(f"Erreur lors de la génération d'image: {str(e)}")
        return json.dumps({"error": f"Erreur: {str(e)}"})