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
import spaces
from io import BytesIO

# Détecter l'environnement (MLX sur Mac M1/M2/M3 ou Hugging Face sur le cloud)
is_mac_mlx = sys.platform == "darwin" and platform.processor() == 'arm'

gpu_timeout = 120

# Récupérer le token depuis les variables d'environnement
hf_token = os.environ.get("hf_token")

# Variable pour indiquer le mode (MLX ou Transformers)
MODEL_MODE = "mlx" if is_mac_mlx else "transformers"
    

# Configuration du modèle selon l'environnement
if MODEL_MODE == "mlx":
    # Version MLX pour Mac Apple Silicon
    import mlx.core as mx
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    MODEL_NAME = "mlx-community/Mistral-Small-3.1-24B-Instruct-2503-8bit"
    spaces.GPU = lambda duration=None: (lambda func: func)
else:
    #MODEL_NAME = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    MODEL_NAME = "ISTA-DASLab/Mistral-Small-3.1-24B-Instruct-2503-GPTQ-4b-128g"
    # IMPORTANT: Pour éviter l'initialisation CUDA dans le processus principal
    # On n'importe pas torch ici au niveau global
    # L'import de torch et autres sera fait uniquement dans les fonctions décorées avec @spaces.GPU()

# Chemins et configurations
TEMP_DIR = tempfile.gettempdir()

# Variable globale pour stocker les données
global_state = {
    "file_paths": [],          # Liste des chemins de tous les fichiers
    "all_sections": [],        # Sections détectées dans les documents
    "image_paths": {},         # Stocke les chemins des images par (doc_index, page)
    "model": None,
    "processor": None,
    "config": None,
    "transformer_model": None, # Pour la version Transformers
    "transformer_processor": None, # Pour la version Transformers
    "device": None,            # Appareil utilisé pour le traitement
    "initialization_failed": False # Indique si l'initialisation a échoué
}

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
@spaces.GPU(duration=gpu_timeout)
def free_memory():
    """Libère la mémoire utilisée par les modèles."""
    if MODEL_MODE == "mlx":
        if "model" in global_state:
            global_state["model"] = None
            global_state["processor"] = None
            global_state["config"] = None
            import gc
            gc.collect()
    else:
        if "transformer_model" in global_state:
            import torch
            global_state["transformer_model"] = None
            global_state["transformer_processor"] = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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

# Fonction pour charger le modèle selon l'environnement
@spaces.GPU(duration=gpu_timeout)
def load_model():
    try:
        if global_state.get("initialization_failed", False):
            print("L'initialisation a déjà échoué précédemment, retour sans modèle")
            return None, None, None, None, None

        if MODEL_MODE == "mlx":
            if global_state["model"] is None:
                print("Chargement du modèle MLX-VLM...")
                model, processor = load(MODEL_NAME)
                config = load_config(MODEL_NAME)
                global_state["model"] = model
                global_state["processor"] = processor
                global_state["config"] = config
                print("Modèle MLX chargé avec succès!")
            return global_state["model"], global_state["processor"], global_state["config"], None, None

        else:
            if global_state["transformer_model"] is None:
                import torch
                from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

                device = "cuda" if torch.cuda.is_available() else "cpu"
                global_state["device"] = device

                print(f"Utilisation de l'appareil: {device}")

                #quantization_config = BitsAndBytesConfig(
                #    load_in_4bit=True,
                #    bnb_4bit_compute_dtype=torch.bfloat16,
                #    bnb_4bit_use_double_quant=True,
                #    bnb_4bit_quant_type="nf4"
                #)

                if not isinstance(MODEL_NAME, str) or not MODEL_NAME:
                    raise ValueError(f"MODEL_NAME invalide : {MODEL_NAME}")

                print(f"Chargement du modèle {MODEL_NAME}...")
                processor = AutoProcessor.from_pretrained(MODEL_NAME,token=hf_token)
                model = AutoModelForImageTextToText.from_pretrained(
                    MODEL_NAME,
                    device_map="auto",
                    #quantization_config=quantization_config,
                    torch_dtype=torch.bfloat16, 
                    #low_cpu_mem_usage=True,
                    token=hf_token  # Utiliser le token ici
                )

                global_state["transformer_model"] = model
                global_state["transformer_processor"] = processor

                print(f"Modèle {MODEL_NAME} chargé avec succès sur {device}!")

            return global_state["model"], global_state["processor"], global_state["config"], global_state["transformer_model"], global_state["transformer_processor"]

    except Exception as e:
        print(f"Erreur critique lors du chargement du modèle: {str(e)}")
        global_state["initialization_failed"] = True
        global_state["model"] = None
        global_state["processor"] = None
        global_state["config"] = None
        global_state["transformer_model"] = None
        global_state["transformer_processor"] = None
        raise e

# Fonction pour générer du texte à partir d'une image selon l'environnement
@spaces.GPU(duration=gpu_timeout)
def generate_from_image(image_path, prompt_text, model=None, processor=None, config=None, transformer_model=None, transformer_processor=None):
    try:
        # Si l'initialisation a échoué, retourner une erreur
        if global_state.get("initialization_failed", False):
            return """{"error": "Le modèle n'a pas pu être chargé. Veuillez vérifier les logs."}"""
            
        # Import ici pour éviter initialisation CUDA dans le processus principal
        if MODEL_MODE != "mlx":
            import torch
        
        if MODEL_MODE == "mlx":
            # Version MLX
            formatted_prompt = apply_chat_template(
                processor, config, prompt_text, num_images=1
            )
            result = generate(model, processor, formatted_prompt, [image_path], verbose=False, max_tokens=2048, temperature=0.1)
            return result
        else:
            try:
                # Charger l'image
                image = Image.open(image_path).convert('RGB')
                
                # Créer le message pour le modèle
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "url": image_path},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ]
                
                # Appliquer le template et tokenizer
                device = global_state.get("device", "cpu")
                inputs = transformer_processor.apply_chat_template(
                    messages, 
                    add_generation_prompt=True, 
                    tokenize=True, 
                    truncation=True,
                    return_dict=True, 
                    return_tensors="pt"
                ).to(device).to(torch.bfloat16)
                
                # Limiter le nombre de tokens selon l'appareil
                max_tokens = 1024 if device == "cpu" else 2048
                
                # Générer la réponse
                with torch.no_grad():
                    try:
                        generate_ids = transformer_model.generate(
                            **inputs, 
                            max_new_tokens=max_tokens,
                            do_sample=True
                        )
                        
                        # Décoder la sortie
                        result = transformer_processor.decode(
                            generate_ids[0, inputs["input_ids"].shape[1]:], 
                            skip_special_tokens=True
                        )
                        
                        return result
                    except Exception as gen_error:
                        print(f"Erreur lors de la génération: {str(gen_error)}")
                        
                        # Essayer avec des paramètres plus conservateurs
                        print("Tentative avec des paramètres réduits...")
                        generate_ids = transformer_model.generate(
                            **inputs, 
                            max_new_tokens=512,  # Réduire le nombre de tokens
                            temperature=0.0,     # Déterministe
                            do_sample=False      # Pas d'échantillonnage
                        )
                        
                        # Décoder la sortie
                        result = transformer_processor.decode(
                            generate_ids[0, inputs["input_ids"].shape[1]:], 
                            skip_special_tokens=True
                        )
                        
                        return result
            except Exception as e:
                print(f"Erreur lors de la génération: {str(e)}")
                return json.dumps({"error": f"Erreur lors de la génération: {str(e)}"})
    except Exception as e:
        print(f"Erreur lors de la génération d'image: {str(e)}")
        return f"""{"error": "Erreur: {str(e)}"}"""

# Fonction pour extraire les titres de section directement à partir de l'image
@spaces.GPU(duration=gpu_timeout)
def extract_sections_from_image(image_path, model=None, processor=None, config=None, transformer_model=None, transformer_processor=None):
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
        result = generate_from_image(image_path, prompt, model, processor, config, transformer_model, transformer_processor)
        
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
@spaces.GPU(duration=gpu_timeout)
def extract_section_values(image_path, selected_sections, model=None, processor=None, config=None, transformer_model=None, transformer_processor=None):
    try:
        # Validation des entrées
        if not image_path or not os.path.exists(image_path):
            return []
            
        if not selected_sections or not isinstance(selected_sections, list):
            return []
            
        # Transformer les sections sélectionnées en texte pour le prompt
        sections_text = "\n".join([f"- {section}" for section in selected_sections])
        
        # Préparation du prompt pour extraire les valeurs des sections
        prompt = f"""
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
        
        # Générer le résultat
        result = generate_from_image(image_path, prompt, model, processor, config, transformer_model, transformer_processor)
        
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

# Fonction pour scanner et extraire les sections d'un document
def magic_scan(file, progress=gr.Progress()):
    try:
        if file is None:
            return gr.update(choices=[]), None
        
        # Vérifier le fichier
        file_path = file.name
        global_state["file_paths"] = [file_path]  # Pour le moment, un seul fichier
        _, file_extension = os.path.splitext(file_path)
        
        # Charger le modèle selon l'environnement
        progress(0.1, "Chargement du modèle...")
        model, processor, config, transformer_model, transformer_processor = load_model()
        
        # Si l'initialisation a échoué et qu'aucun modèle n'est disponible
        if global_state.get("initialization_failed", False) or (MODEL_MODE != "mlx" and transformer_model is None):
            return gr.update(choices=["Erreur: Le modèle n'a pas pu être chargé. Voir les logs pour plus de détails."]), None
        
        # Réinitialiser les chemins d'image et nettoyer les anciens fichiers
        cleanup_temp_files()
        global_state["image_paths"] = {}
        
        # Traiter selon le type de fichier
        all_sections = []
        
        if file_extension.lower() == '.pdf':
            progress(0.2, "Conversion du PDF en images...")
            # Convertir PDF en images
            images = pdf2image.convert_from_path(file_path)
            
            # Traiter jusqu'à 5 premières pages pour plus de couverture
            for i, img in enumerate(images[:5]):
                progress(0.3 + (i * 0.1), f"Analyse de la page {i+1}/{min(5, len(images))}...")
                
                # Sauvegarder temporairement l'image
                temp_img_path = os.path.join(TEMP_DIR, f"temp_doc0_page_{i}.png")
                img.save(temp_img_path, "PNG", quality=95, dpi=(300, 300))
                
                # Stocker le chemin de l'image pour utilisation ultérieure
                global_state["image_paths"][(0, i+1)] = temp_img_path
                
                # Traiter l'image avec le modèle
                page_sections = extract_sections_from_image(temp_img_path, model, processor, config, transformer_model, transformer_processor)
                
                # Ajouter les titres trouvés avec indication de la page
                for section in page_sections:
                    section["page"] = i + 1
                    section["doc_index"] = 0
                    all_sections.append(section)
        else:
            progress(0.4, "Analyse de l'image...")
            # Pour les images directement
            global_state["image_paths"][(0, 1)] = file_path
            all_sections = extract_sections_from_image(file_path, model, processor, config, transformer_model, transformer_processor)
            for section in all_sections:
                section["page"] = 1
                section["doc_index"] = 0
        
        progress(0.8, "Finalisation des résultats...")
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
        
        progress(1.0, "Analyse terminée!")
        return gr.update(choices=section_choices, value=[]), json_path
    except Exception as e:
        print(f"Erreur lors du scan: {str(e)}")
        return gr.update(choices=[f"Erreur: {str(e)}"]), None

# Fonction pour ajouter des sections manuellement
def add_manual_sections(section_names, existing_sections):
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

# Fonction pour traiter plusieurs documents
def process_multiple_documents(files, selected_sections, progress=gr.Progress()):
    try:
        if not files or not selected_sections:
            return "Veuillez sélectionner des documents et des sections à extraire.", None
        
        # Charger le modèle
        progress(0.1, "Chargement du modèle...")
        model, processor, config, transformer_model, transformer_processor = load_model()
        
        # Si l'initialisation a échoué et qu'aucun modèle n'est disponible
        if global_state.get("initialization_failed", False) or (MODEL_MODE != "mlx" and transformer_model is None):
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
        
        # Calculer la distribution de la progression
        progress_step = 0.8 / max(1, len(files))
        
        # Traiter chaque fichier
        for doc_index, file in enumerate(files):
            progress(0.1 + (doc_index * progress_step), f"Traitement du document {doc_index+1}/{len(files)}...")
            
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
                
                # Calculer la distribution de la progression pour les pages
                page_progress_step = progress_step / max(1, min(5, len(images)))
                
                # Traiter jusqu'à 5 premières pages
                for page_num, img in enumerate(images[:5]):
                    page_progress = 0.1 + (doc_index * progress_step) + (page_num * page_progress_step)
                    progress(page_progress, f"Document {doc_index+1}/{len(files)}, page {page_num+1}/{min(5, len(images))}...")
                    
                    # Sauvegarder temporairement l'image
                    temp_img_path = os.path.join(TEMP_DIR, f"temp_doc{doc_index}_page_{page_num}.png")
                    img.save(temp_img_path, "PNG", quality=95, dpi=(300, 300))
                    
                    # Extraire les valeurs des sections pour cette page
                    page_values = extract_section_values(temp_img_path, section_titles, model, processor, config, transformer_model, transformer_processor)
                    
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
                page_values = extract_section_values(file_path, section_titles, model, processor, config, transformer_model, transformer_processor)
                
                # Ajouter le numéro de page
                for value in page_values:
                    value["page"] = 1
                
                # Ajouter à la liste des valeurs extraites
                doc_result["extracted_values"].extend(page_values)
            
            # Ajouter les résultats de ce document
            all_results.append(doc_result)
        
        # Organiser les résultats
        progress(0.9, "Finalisation des résultats...")
        result = {
            "timestamp": import_time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": MODEL_MODE,  # Indiquer si c'est la version MLX ou Transformers
            "documents": all_results
        }
        
        # Sauvegarder en JSON
        json_path = os.path.join(TEMP_DIR, "multi_doc_extracted_values.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # Préparer un résumé pour l'affichage
        summary = f"Extraction réussie pour {len(files)} documents (Mode: {MODEL_MODE}):\n\n"
        
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
        
        progress(1.0, "Extraction terminée!")
        return summary, json_path
    except Exception as e:
        print(f"Erreur lors du traitement multiple: {str(e)}")
        # Nettoyer en cas d'erreur
        cleanup_temp_files()
        free_memory()
        return f"Erreur lors de l'extraction: {str(e)}", None

# Fonction pour afficher l'info sur le mode d'exécution actuel
def show_runtime_info():
    if MODEL_MODE == "mlx":
        return "Exécution avec MLX sur Mac Apple Silicon - Modèle: mlx-community/Mistral-Small-3.1-24B-Instruct-2503-8bit"
    else:
        device = global_state.get("device", "inconnu")
        if global_state.get("initialization_failed", False):
            return f"ERREUR: Le modèle n'a pas pu être chargé. Consultez les logs pour plus de détails."
        else:
            return f"Exécution avec Transformers sur {device} - Modèle: {MODEL_NAME}"

if MODEL_MODE == "transformers":
    @spaces.GPU(duration=gpu_timeout)
    def gpu_magic_scan(file, progress=gr.Progress()):
        return magic_scan(file, progress)

    @spaces.GPU(duration=gpu_timeout)
    def gpu_process_multiple_documents(files, selected_sections, progress=gr.Progress()):
        return process_multiple_documents(files, selected_sections, progress)
else:  # MLX
    gpu_magic_scan = magic_scan
    gpu_process_multiple_documents = process_multiple_documents

# Interface Gradio
with gr.Blocks(title="Magic Document Scanner") as app:
    gr.Markdown("# 📄 Magic Document Scanner")
    runtime_info = gr.Markdown(show_runtime_info())
    
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
        
        extract_button = gr.Button("✨ Extraire les valeurs des sections sélectionnées", variant="primary")
        extraction_results = gr.Textbox(label="Résultats de l'extraction", lines=15)
        values_json_output = gr.File(label="Résultat JSON des valeurs extraites")
    
    # Connecter les événements

    scan_button.click(
        fn=gpu_magic_scan,
        inputs=file_input,
        outputs=[sections_output, json_output]
    )
    
    add_sections_button.click(
        fn=add_manual_sections,
        inputs=[manual_sections, sections_output],
        outputs=sections_output
    )
    
    extract_button.click(
        fn=gpu_process_multiple_documents,
        inputs=[files_input, sections_output],
        outputs=[extraction_results, values_json_output]
    )
    
    gr.Markdown("""
    ### Mode d'emploi :
    1. **Configurer les sections** : 
       - Téléchargez un document modèle et cliquez sur "Magic Scan" pour détecter automatiquement les sections.
       - OU ajoutez manuellement des sections en les saisissant (une par ligne) et en cliquant sur "Ajouter ces sections".
       - Cochez les sections que vous souhaitez extraire.
    
    2. **Extraire les valeurs** :
       - Téléchargez un ou plusieurs documents du même type.
       - Cliquez sur "Extraire les valeurs" pour obtenir les informations des sections sélectionnées.
       - Les résultats sont disponibles au format JSON.
       
    ### Informations techniques :
    - **Mac Apple Silicon** : utilisation de MLX pour un traitement local optimisé
    - **Autres plateformes** : utilisation de Mistral-Small-3.1-24B-Instruct (modèle quantifié 4-bit)
    - Compatible avec les environnements GPU stateless de Hugging Face Spaces
    - Traitement des PDFs et des images
    """)

# Lancer l'application
if __name__ == "__main__":
    app.launch()