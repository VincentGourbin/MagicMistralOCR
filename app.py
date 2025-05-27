import os
import json
import gradio as gr
import time
import platform
import sys

# Importer les modules du projet
from config import global_state, TEMP_DIR, show_runtime_info
from utils import cleanup_temp_files, free_memory
from model_handler import load_model
from api_client import update_api_config
from image_processor import convert_pdf_to_images
from data_extractor import extract_sections_from_image, extract_section_values
from mcp_functions import analyze_document, extract_values

# Fonction pour afficher simplement le mode actuel
def display_current_mode():
    """Affiche simplement le mode actuel"""
    mode = global_state["MODE"]
    if mode == "api":
        api_config = global_state["api_config"]
        server = api_config["server"]
        model = api_config["model"]
        key_preview = api_config["api_key"][:4] + "..." if api_config["api_key"] and len(api_config["api_key"]) > 4 else "Non définie"
        return f"### Mode actuel : API EXTERNE\n- Serveur : {server}\n- Modèle : {model}\n- Clé : {key_preview}"
    else:
        return f"### Mode actuel : MLX LOCAL\n- Modèle : {global_state['MODEL_NAME']}"

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
        global_state["file_paths"] = [file_path]
        _, file_extension = os.path.splitext(file_path)
        
        print(f"*** MAGIC SCAN: MODE ACTUEL = {global_state['MODE']} ***")
        
        # Charger le modèle selon l'environnement
        model, processor, config = load_model()
        
        # Si l'initialisation a échoué et qu'aucun modèle n'est disponible (sauf en mode API)
        if global_state.get("initialization_failed", False) and global_state["MODE"] != "api":
            return gr.update(choices=["Erreur: Le modèle n'a pas pu être chargé. Voir les logs pour plus de détails."]), None
        
        # Réinitialiser les chemins d'image et nettoyer les anciens fichiers
        cleanup_temp_files()
        global_state["image_paths"] = {}
        
        # Traiter selon le type de fichier
        all_sections = []
        
        if file_extension.lower() == '.pdf':
            # Convertir PDF en images
            from image_processor import convert_pdf_to_images
            img_paths = convert_pdf_to_images(file_path, page_limit=5, doc_index=0)
            
            # Traiter chaque image convertie
            for i, img_path in enumerate(img_paths):
                # Traiter l'image avec le modèle
                from data_extractor import extract_sections_from_image
                page_sections = extract_sections_from_image(img_path, model, processor, config)
                
                # Ajouter les titres trouvés avec indication de la page
                for section in page_sections:
                    section["page"] = i + 1
                    section["doc_index"] = 0
                    all_sections.append(section)
        else:
            # Pour les images directement
            global_state["image_paths"][(0, 1)] = file_path
            from data_extractor import extract_sections_from_image
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
def add_manual_sections(section_names, existing_choices=None):
    """
    Ajoute des sections définies manuellement à la liste des sections existantes.
    
    Args:
        section_names (str): Texte contenant les noms des sections à ajouter, une par ligne
        existing_choices (list, optional): Liste des choix existants (ignorée, pour compatibilité)
        
    Returns:
        gr.update: Mise à jour du composant CheckboxGroup avec toutes les sections
    """    
    try:
        # Validation des entrées
        if not isinstance(section_names, str) or not section_names.strip():
            return gr.update()
        
        # Utiliser les sections déjà présentes dans l'état global
        existing_sections = global_state.get("all_sections", [])
        
        # Extraire tous les titres existants pour éviter les doublons
        existing_titles = {section.get("title", "") for section in existing_sections}
        
        print(f"Titres existants: {existing_titles}")
        
        # Traiter les nouvelles sections (une par ligne)
        for section in section_names.split('\n'):
            section = section.strip()
            if section and len(section) <= 100 and section not in existing_titles:  # Limite raisonnable
                # Ajouter à l'état global
                global_state["all_sections"].append({
                    "title": section,
                    "level": 1,
                    "type": "manual",
                    "page": 1,
                    "doc_index": -1  # Indique une entrée manuelle
                })
                existing_titles.add(section)
        
        # Préparer les choix pour le composant CheckboxGroup à partir de l'état global mis à jour
        updated_sections = global_state.get("all_sections", [])
        section_choices = [
            f"{section['title']} (Niveau: {section.get('level', 1)}, Type: {section.get('type', 'section')}, Page: {section.get('page', 1)})"
            for section in updated_sections
        ]
        
        print(f"Sections combinées: {section_choices}")
        
        # Retourner les nouvelles sections combinées
        return gr.update(choices=section_choices)
    except Exception as e:
        print(f"Erreur lors de l'ajout de sections manuelles: {str(e)}")
        return gr.update()

# Fonction pour mettre à jour l'affichage des sections sélectionnées
def update_selected_sections_display(selected_sections):
    """
    Met à jour l'affichage des sections sélectionnées dans l'onglet d'extraction.
    
    Args:
        selected_sections (list): Liste des sections sélectionnées dans l'interface
        
    Returns:
        str: Texte formaté des sections sélectionnées
    """
    try:
        if not selected_sections or not isinstance(selected_sections, list):
            return "Aucune section sélectionnée"
        
        # Extraire uniquement les noms des sections sans les informations supplémentaires
        section_names = []
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
                
            section_names.append(title)
        
        # Joindre les noms avec des virgules
        return ", ".join(section_names)
    except Exception as e:
        print(f"Erreur lors de la mise à jour de l'affichage des sections: {str(e)}")
        return "Erreur lors de la récupération des sections"

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
        current_mode = global_state["MODE"]
        if global_state.get("initialization_failed", False) and current_mode != "api":
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
        temp_img_paths = []
        
        # Traiter chaque fichier
        for doc_index, file in enumerate(files):
            file_path = file.name
            _, file_extension = os.path.splitext(file_path)
            is_pdf = file_extension.lower() == '.pdf'
            
            # Résultat pour ce document
            doc_result = {
                "document": os.path.basename(file_path),
                "extracted_values": []
            }
            
            # Traiter selon le type de fichier (identique pour les deux modes)
            if is_pdf:
                # Convertir PDF en images
                from image_processor import convert_pdf_to_images
                img_paths = convert_pdf_to_images(file_path, doc_index=doc_index)
                temp_img_paths.extend(img_paths)
                
                # Traiter chaque image
                for page_num, img_path in enumerate(img_paths):
                    # Extraire les valeurs des sections pour cette page
                    expert_text = expert_prompt if expert_mode_enabled else ""
                    from data_extractor import extract_section_values
                    page_values = extract_section_values(img_path, section_titles, model, processor, config, expert_text)
                    
                    # Ajouter le numéro de page à chaque résultat
                    for value in page_values:
                        value["page"] = page_num + 1
                    
                    # Ajouter à la liste des valeurs extraites
                    doc_result["extracted_values"].extend(page_values)
            else:
                # Pour les images directement
                expert_text = expert_prompt if expert_mode_enabled else ""
                from data_extractor import extract_section_values
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
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": current_mode,  # Indiquer le mode utilisé
            "expert_mode": expert_mode_enabled,
            "documents": all_results
        }
        
        # Sauvegarder en JSON
        json_path = os.path.join(TEMP_DIR, "multi_doc_extracted_values.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # Préparer un résumé pour l'affichage
        summary = f"Extraction réussie pour {len(files)} documents (Mode: {current_mode}"
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
        
        return summary, json_path
    except Exception as e:
        print(f"Erreur lors du traitement multiple: {str(e)}")
        return f"Erreur lors de l'extraction: {str(e)}", None
    finally:
        # Nettoyer les fichiers temporaires
        for path in temp_img_paths:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"Erreur lors du nettoyage de l'image temporaire: {str(e)}")
        
        # Nettoyer les autres fichiers temporaires
        cleanup_temp_files()
        
        # Libérer la mémoire du modèle si possible
        free_memory()

# Fonction pour mettre à jour la configuration API directement dans l'interface
def direct_api_config_update(server, model, api_key, enabled):
    """Fonction qui met à jour la configuration API et force la mise à jour de l'interface"""
    result = update_api_config(server, model, api_key, enabled)
    # Renvoyer aussi l'état actuel du mode pour l'affichage
    return result, display_current_mode()

# Exposer les fonctions MCP pour les assistants IA
analyze_document_for_mcp = analyze_document
extract_values_from_document = extract_values

# Interface Gradio
with gr.Blocks(title="Magic Document Scanner") as app:
    gr.Markdown("# 📄 Magic Document Scanner")
    
    # Affichage du mode actuel (remplace l'ancien bandeau)
    current_mode_display = gr.Markdown(display_current_mode())
    
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
                update_api_button = gr.Button("💾 Sauvegarder et activer", variant="primary")
    
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
        
        # Affichage des sections sélectionnées
        selected_sections_display = gr.Textbox(
            label="Sections sélectionnées",
            placeholder="Les sections sélectionnées dans l'onglet précédent apparaîtront ici",
            interactive=False
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
        fn=direct_api_config_update,
        inputs=[api_server, api_model, api_key, api_enabled],
        outputs=[gr.Markdown(), current_mode_display]
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
    
    # Mettre à jour l'affichage des sections sélectionnées lorsqu'elles changent
    sections_output.change(
        fn=update_selected_sections_display,
        inputs=sections_output,
        outputs=selected_sections_display
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
       - Cliquez sur "Sauvegarder et activer"
       
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
    """)

# Lancer l'application
if __name__ == "__main__":
    app.launch(mcp_server=True)  # Activer le serveur MCP