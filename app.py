import gradio as gr
import os
import sys
import tempfile
import json
from PIL import Image
import pdf2image
import time as import_time
import spaces
from io import BytesIO

# Détecter l'environnement (MLX sur Mac M1/M2/M3 ou Hugging Face sur le cloud)
is_mac_mlx = sys.platform == "darwin" and os.path.exists("/proc/cpuinfo") == False

gpu_timeout = int(os.getenv("GPU_TIMEOUT", 60))

# Variable pour indiquer le mode (MLX ou HF)
MODEL_MODE = "mlx" if is_mac_mlx else "vllm"

# Configuration du modèle selon l'environnement
if MODEL_MODE == "mlx":
    # Version MLX pour Mac Apple Silicon
    import mlx.core as mx
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    MODEL_NAME = "mlx-community/Mistral-Small-3.1-24B-Instruct-2503-8bit"
else:
    # Version VLLM pour Hugging Face Spaces avec les paramètres corrects
    import torch
    from vllm import LLM
    from vllm.sampling_params import SamplingParams
    from vllm.inputs.data import TokensPrompt
    from vllm.multimodal import MultiModalDataBuiltins
    
    from mistral_common.protocol.instruct.messages import TextChunk, ImageURLChunk
    
    MODEL_NAME = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

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
    "llm": None,               # Pour la version VLLM
    "sampling_params": None    # Pour la version VLLM
}

# Fonction pour charger le modèle selon l'environnement
@spaces.GPU(duration=gpu_timeout)
def load_model():
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
        if global_state["llm"] is None:
            print("Chargement du modèle avec VLLM...")
            # Utiliser les paramètres corrects mentionnés sur le forum
            llm = LLM(
                model=MODEL_NAME,
                tokenizer_mode="mistral",  # Obligatoire pour Mistral
                config_format="mistral",   # Paramètre clé pour Mistral-3.1
                load_format="mistral",     # Paramètre clé pour Mistral-3.1
                dtype="float16" if torch.cuda.is_available() else "float32",
                gpu_memory_utilization=0.8
            )
            
            # Paramètres d'échantillonnage pour la génération
            sampling_params = SamplingParams(
                max_tokens=2048,
                temperature=0.1
            )
            
            global_state["llm"] = llm
            global_state["sampling_params"] = sampling_params
            print(f"Modèle VLLM chargé avec succès!")
        return None, None, None, global_state["llm"], global_state["sampling_params"]

# Fonction pour générer du texte à partir d'une image selon l'environnement
def generate_from_image(image_path, prompt_text, model=None, processor=None, config=None, llm=None, sampling_params=None):
    if MODEL_MODE == "mlx":
        # Version MLX
        formatted_prompt = apply_chat_template(
            processor, config, prompt_text, num_images=1
        )
        result = generate(model, processor, formatted_prompt, [image_path], verbose=False, max_tokens=2048, temperature=0.1)
        return result
    else:
        # Version VLLM - suivre l'exemple adapté avec les paramètres corrects
        try:
            # Charger l'image depuis le chemin
            image = Image.open(image_path)
            
            # Créer un URL factice pour l'image locale
            dummy_url = f"file://{image_path}"
            
            # Créer le contenu utilisateur
            user_content = [ImageURLChunk(image_url=dummy_url), TextChunk(text=prompt_text)]
            
            # Obtenir le tokenizer de Mistral
            tokenizer = llm.llm_engine.tokenizer.tokenizer.mistral.instruct_tokenizer
            
            # Encoder le contenu utilisateur
            tokens = tokenizer.encode_user_content(user_content, False)[0]
            
            # Créer le prompt avec tokens et image
            vllm_prompt = TokensPrompt(
                prompt_token_ids=tokens,
                multi_modal_data=MultiModalDataBuiltins(image=[image])
            )
            
            # Générer la réponse
            outputs = llm.generate(vllm_prompt, sampling_params=sampling_params)
            result = outputs[0].outputs[0].text
            
            return result
        except Exception as e:
            print(f"Erreur lors de la génération VLLM: {str(e)}")
            # En cas d'erreur, essayer une approche alternative pour le débogage
            return f"Erreur VLLM: {str(e)}"

# Fonction pour extraire les titres de section directement à partir de l'image
def extract_sections_from_image(image_path, model=None, processor=None, config=None, llm=None, sampling_params=None):
    # Préparation du prompt pour le modèle VLM avec demande explicite de format JSON
    prompt = """
    Examine cette image de document et extrait tous les titres de sections, champs ou entités présentes.
    
    Retourne UNIQUEMENT une liste au format JSON sous cette forme précise :
    {
      "sections": [
        {
          "title": "Nom du titre ou champ 1",
          "level": 1,
          "type": "section|field|header"
        },
        {
          "title": "Nom du titre ou champ 2",
          "level": 2,
          "type": "section|field|header"
        }
      ]
    }
    
    Assure-toi que le JSON est parfaitement formaté et inclut absolument TOUTES les sections ou champs visibles.
    Ne renvoie aucune explication, juste le JSON.
    """
    
    # Générer le résultat
    result = generate_from_image(image_path, prompt, model, processor, config, llm, sampling_params)
    
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

# Fonction pour extraire les valeurs des sections sélectionnées
def extract_section_values(image_path, selected_sections, model=None, processor=None, config=None, llm=None, sampling_params=None):
    # Transformer les sections sélectionnées en texte pour le prompt
    sections_text = "\n".join([f"- {section}" for section in selected_sections])
    
    # Préparation du prompt pour extraire les valeurs des sections
    prompt = f"""
    Examine cette image de document et extrait les valeurs correspondant exactement aux champs ou sections suivants:
    
    {sections_text}
    
    Pour chaque section ou champ, trouve la valeur ou le contenu correspondant.
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
        }}
      ]
    }}
    
    Si tu ne trouves pas de valeur pour un champ, indique une chaîne vide pour "value" et 0 pour "confidence".
    Assure-toi que le JSON est parfaitement formaté. Ne renvoie aucune explication, juste le JSON.
    """
    
    # Générer le résultat
    result = generate_from_image(image_path, prompt, model, processor, config, llm, sampling_params)
    
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

# Fonction pour scanner et extraire les sections d'un document
def magic_scan(file):
    if file is None:
        return gr.update(choices=[]), None
    
    # Vérifier le fichier
    file_path = file.name
    global_state["file_paths"] = [file_path]  # Pour le moment, un seul fichier
    _, file_extension = os.path.splitext(file_path)
    
    # Charger le modèle selon l'environnement
    model, processor, config, llm, sampling_params = load_model()
    
    # Réinitialiser les chemins d'image
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
            page_sections = extract_sections_from_image(temp_img_path, model, processor, config, llm, sampling_params)
            
            # Ajouter les titres trouvés avec indication de la page
            for section in page_sections:
                section["page"] = i + 1
                section["doc_index"] = 0
                all_sections.append(section)
    else:
        # Pour les images directement
        global_state["image_paths"][(0, 1)] = file_path
        all_sections = extract_sections_from_image(file_path, model, processor, config, llm, sampling_params)
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

# Fonction pour ajouter des sections manuellement
def add_manual_sections(section_names, existing_sections):
    if not section_names.strip():
        return gr.update()
    
    # Récupérer les sections existantes
    current_choices = existing_sections.choices if hasattr(existing_sections, 'choices') else []
    current_values = existing_sections.value if hasattr(existing_sections, 'value') else []
    
    # Traiter les nouvelles sections (une par ligne)
    new_sections = [s.strip() for s in section_names.split('\n') if s.strip()]
    
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

# Fonction pour traiter plusieurs documents
def process_multiple_documents(files, selected_sections):
    if not files or not selected_sections:
        return "Veuillez sélectionner des documents et des sections à extraire.", None
    
    # Charger le modèle
    model, processor, config, llm, sampling_params = load_model()
    
    # Extraire les titres des sections sélectionnées
    section_titles = []
    for section in selected_sections:
        # Extraire le titre (tout ce qui est avant "(Niveau:")
        if " (Niveau:" in section:
            title = section.split(" (Niveau:")[0]
        else:
            title = section
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
                page_values = extract_section_values(temp_img_path, section_titles, model, processor, config, llm, sampling_params)
                
                # Ajouter le numéro de page à chaque résultat
                for value in page_values:
                    value["page"] = page_num + 1
                
                # Ajouter à la liste des valeurs extraites
                doc_result["extracted_values"].extend(page_values)
                
                # Nettoyage
                os.remove(temp_img_path)
        else:
            # Pour les images directement
            page_values = extract_section_values(file_path, section_titles, model, processor, config, llm, sampling_params)
            
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
        "mode": MODEL_MODE,  # Indiquer si c'est la version MLX ou VLLM
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
    
    return summary, json_path

# Fonction pour afficher l'info sur le mode d'exécution actuel
def show_runtime_info():
    if MODEL_MODE == "mlx":
        return "Exécution avec MLX sur Mac Apple Silicon - Modèle: mlx-community/Mistral-Small-3.1-24B-Instruct-2503-8bit"
    else:
        device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        return f"Exécution avec VLLM sur {device} - Modèle: mistralai/Mistral-Small-3.1-24B-Instruct-2503"

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
    """)

# Lancer l'application
if __name__ == "__main__":
    app.launch()