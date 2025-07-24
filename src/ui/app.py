"""
Magic Document Scanner - Interface utilisateur Gradio

Application de traitement intelligent de documents avec OCR et extraction de données.
Interface web basée sur Gradio avec parallélisation optimisée pour les APIs.

Features:
- Scan automatique des sections de documents
- Extraction de données avec mode expert
- Support MLX (local) et API externes
- Parallélisation intelligente configurée
- Filtrage avancé des pages
"""

import os
import json
import gradio as gr
import time
import platform
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# Importer les modules du projet
from core.config import global_state, TEMP_DIR, show_runtime_info
from utils.utils import cleanup_temp_files, free_memory
from core.model_handler import load_model
from core.api_client import update_api_config
from utils.image_processor import convert_pdf_to_images
from core.data_extractor import extract_sections_from_image, extract_section_values, should_process_page
from core.mcp_functions import analyze_document, extract_values

# Fonction pour afficher simplement le mode actuel
def display_current_mode():
    """
    Affiche le mode d'exécution actuel avec ses paramètres.
    
    Returns:
        str: Texte formaté décrivant le mode actuel (API/MLX) avec ses paramètres
    """
    mode = global_state["MODE"]
    if mode == "api":
        api_config = global_state["api_config"]
        server = api_config["server"]
        model = api_config["model"]
        pool_size = api_config.get("pool_size", 5)
        key_preview = api_config["api_key"][:4] + "..." if api_config["api_key"] and len(api_config["api_key"]) > 4 else "Non définie"
        return f"### Mode actuel : API EXTERNE\n- Serveur : {server}\n- Modèle : {model}\n- Pool : {pool_size} threads\n- Clé : {key_preview}"
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

# Fonction pour obtenir la taille du pool selon le mode
def get_pool_size():
    """
    Retourne la taille du pool de threads selon le mode d'exécution.
    
    La taille du pool détermine le nombre de threads simultanés pour le traitement :
    - Mode API : taille configurable par l'utilisateur (1-20 threads)
    - Mode MLX : forcé à 1 thread (traitement séquentiel)
    
    Returns:
        int: Nombre de threads à utiliser pour la parallélisation
    """
    current_mode = global_state["MODE"]
    if current_mode == "api":
        return global_state["api_config"]["pool_size"]
    else:
        # MLX : pool de taille 1 pour simuler le comportement séquentiel
        return 1

# Fonction pour traitement unifié en parallèle (tous modes)
def process_documents_unified(files, section_titles, expert_prompt, page_include_filter, page_exclude_filter, expert_mode_enabled, model, processor, config, current_mode, temp_img_paths, start_time):
    """
    Traitement unifié en deux phases pour tous les modes :
    - Mode API : pool configurable (parallélisation)
    - Mode MLX : pool de taille 1 (séquentiel)
    Phase 1 : Routage avec pool
    Phase 2 : Extraction avec pool
    """
    try:
        pool_size = get_pool_size()
        mode_desc = f"parallélisation ({pool_size} threads)" if pool_size > 1 else "séquentiel (1 thread)"
        print(f"🚀 Mode {current_mode}: traitement en 2 phases pour {len(files)} document(s) - {mode_desc}")
        
        # Phase 1: Collecter toutes les pages de tous les documents
        all_page_data = []
        all_results = []
        
        for doc_index, file in enumerate(files):
            file_path = file.name
            _, file_extension = os.path.splitext(file_path)
            is_pdf = file_extension.lower() == '.pdf'
            
            # Préparer la structure de résultat pour ce document
            doc_result = {
                "document": os.path.basename(file_path),
                "extracted_values": []
            }
            all_results.append(doc_result)
            
            if is_pdf:
                # Convertir PDF en images
                from image_processor import convert_pdf_to_images
                img_paths = convert_pdf_to_images(file_path, doc_index=doc_index)
                temp_img_paths.extend(img_paths)
                
                # Ajouter chaque page à la liste globale
                for page_num, img_path in enumerate(img_paths):
                    page_data = {
                        "img_path": img_path,
                        "page_num": page_num + 1,
                        "doc_name": os.path.basename(file_path),
                        "doc_index": doc_index,
                        "section_titles": section_titles,
                        "expert_prompt": expert_prompt,
                        "page_include_filter": page_include_filter,
                        "page_exclude_filter": page_exclude_filter,
                        "expert_mode_enabled": expert_mode_enabled,
                        "model": model,
                        "processor": processor,
                        "config": config
                    }
                    all_page_data.append(page_data)
            else:
                # Pour les images individuelles
                page_data = {
                    "img_path": file_path,
                    "page_num": 1,
                    "doc_name": os.path.basename(file_path),
                    "doc_index": doc_index,
                    "section_titles": section_titles,
                    "expert_prompt": expert_prompt,
                    "page_include_filter": page_include_filter,
                    "page_exclude_filter": page_exclude_filter,
                    "expert_mode_enabled": expert_mode_enabled,
                    "model": model,
                    "processor": processor,
                    "config": config
                }
                all_page_data.append(page_data)
        
        total_pages = len(all_page_data)
        print(f"📊 Total pages collectées: {total_pages}")
        
        # Phase 1: Routage avec pool (si mode expert avec filtres)
        pages_to_extract = []
        if expert_mode_enabled and (page_include_filter.strip() or page_exclude_filter.strip()):
            print(f"🔍 PHASE 1: Routage de {total_pages} pages (pool: {pool_size})")
            
            # Réduire le pool si on a beaucoup de pages pour éviter la surcharge API
            routing_pool_size = min(pool_size, 3) if current_mode == "api" and total_pages > 5 else pool_size
            if routing_pool_size != pool_size:
                print(f"🔄 Pool réduit à {routing_pool_size} pour le routage (éviter surcharge API)")
            
            with ThreadPoolExecutor(max_workers=routing_pool_size) as executor:
                future_to_page = {executor.submit(route_single_page, page_data): page_data for page_data in all_page_data}
                
                api_errors = 0
                for future in as_completed(future_to_page):
                    result = future.result()
                    if result["success"] and result["should_process"]:
                        pages_to_extract.append(result["page_data"])
                        print(f"✅ Page {result['page_num']} de {result['doc_name']}: sera traitée")
                    elif result["success"]:
                        print(f"❌ Page {result['page_num']} de {result['doc_name']}: sera ignorée")
                    else:
                        api_errors += 1
                        print(f"💥 Erreur routage page {result['page_num']} de {result['doc_name']}")
                
                if api_errors > 0:
                    print(f"⚠️  {api_errors} erreurs API détectées lors du routage")
            
            print(f"📋 Pages à extraire après routage: {len(pages_to_extract)}/{total_pages}")
        else:
            # Pas de filtrage, toutes les pages seront extraites
            pages_to_extract = all_page_data
            print(f"📋 Pas de filtrage: {len(pages_to_extract)} pages à extraire")
        
        # Phase 2: Extraction avec pool des pages validées
        if pages_to_extract:
            print(f"⚡ PHASE 2: Extraction de {len(pages_to_extract)} pages (pool: {pool_size})")
            
            with ThreadPoolExecutor(max_workers=pool_size) as executor:
                future_to_page = {executor.submit(extract_single_page, page_data): page_data for page_data in pages_to_extract}
                
                for future in as_completed(future_to_page):
                    result = future.result()
                    if result["success"]:
                        # Ajouter les résultats au bon document
                        doc_result = all_results[result["doc_index"]]
                        doc_result["extracted_values"].extend(result["values"])
                        print(f"✅ Extraction page {result['page_num']} de {result['doc_name']} terminée")
                    else:
                        print(f"❌ Erreur extraction page {result['page_num']} de {result['doc_name']}: {result.get('error', 'Erreur inconnue')}")
        
        # Organiser les résultats finaux
        result = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": current_mode,
            "expert_mode": expert_mode_enabled,
            "documents": all_results
        }
        
        # Sauvegarder en JSON
        json_path = os.path.join(TEMP_DIR, "multi_doc_extracted_values.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # Préparer un résumé pour l'affichage
        total_extracted = sum(len(doc["extracted_values"]) for doc in all_results)
        summary = f"Extraction réussie pour {len(files)} documents (Mode: {current_mode}"
        if expert_mode_enabled:
            summary += ", Mode Expert activé"
        summary += f", {total_extracted} valeurs extraites):\n\n"
        
        for doc_result in all_results:
            doc_name = doc_result["document"]
            values = doc_result["extracted_values"]
            summary += f"Document: {doc_name} ({len(values)} valeurs extraites)\n"
            
            for item in values:
                confidence = item.get("confidence", 0) * 100
                page = item.get("page", 1)
                summary += f"  • {item['section']} (p.{page}): {item['value']} (confiance: {confidence:.0f}%)\n"
            
            summary += "\n"
        
        # Calculer et afficher le temps total de traitement
        end_time = time.time()
        total_time = end_time - start_time
        print(f"✅ FIN DE L'EXTRACTION - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  TEMPS TOTAL DE TRAITEMENT: {total_time:.2f} secondes ({total_time/60:.1f} minutes)")
        
        return summary, json_path
        
    except Exception as e:
        # Calculer le temps même en cas d'erreur
        end_time = time.time()
        total_time = end_time - start_time
        print(f"❌ ERREUR LORS DE L'EXTRACTION - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  TEMPS AVANT ERREUR: {total_time:.2f} secondes ({total_time/60:.1f} minutes)")
        print(f"Erreur lors du traitement: {str(e)}")
        return f"Erreur lors de l'extraction: {str(e)}", None

# Fonction pour effectuer le routage d'une page (phase 1)
def route_single_page(page_data):
    """
    Effectue uniquement le routage d'une page (pour optimisation parallèle).
    
    Args:
        page_data (dict): Dictionnaire contenant les informations de la page
        
    Returns:
        dict: Résultat du routage pour cette page
    """
    try:
        img_path = page_data["img_path"]
        page_num = page_data["page_num"]
        doc_name = page_data["doc_name"]
        page_include_filter = page_data["page_include_filter"]
        page_exclude_filter = page_data["page_exclude_filter"]
        model = page_data["model"]
        processor = page_data["processor"]
        config = page_data["config"]
        doc_index = page_data["doc_index"]
        
        # Vérifier si la page doit être traitée (filtrage en mode expert)
        should_process = True
        if page_include_filter.strip() or page_exclude_filter.strip():
            should_process = should_process_page(
                img_path, 
                page_include_filter.strip(), 
                page_exclude_filter.strip(), 
                model, processor, config
            )
        
        return {
            "success": True,
            "page_num": page_num,
            "doc_name": doc_name,
            "doc_index": doc_index,
            "img_path": img_path,
            "should_process": should_process,
            "page_data": page_data  # Garder les données originales pour l'extraction
        }
        
    except Exception as e:
        print(f"Erreur lors du routage de la page {page_num} de {doc_name}: {str(e)}")
        return {
            "success": False,
            "page_num": page_num,
            "doc_name": doc_name,
            "doc_index": doc_index,
            "error": str(e),
            "should_process": False
        }

# Fonction pour extraire les données d'une page (phase 2)
def extract_single_page(page_data):
    """
    Effectue uniquement l'extraction d'une page (après validation du routage).
    
    Args:
        page_data (dict): Dictionnaire contenant les informations de la page
        
    Returns:
        dict: Résultats de l'extraction pour cette page
    """
    try:
        img_path = page_data["img_path"]
        page_num = page_data["page_num"]
        doc_name = page_data["doc_name"]
        doc_index = page_data["doc_index"]
        section_titles = page_data["section_titles"]
        expert_prompt = page_data["expert_prompt"]
        expert_mode_enabled = page_data["expert_mode_enabled"]
        model = page_data["model"]
        processor = page_data["processor"]
        config = page_data["config"]
        
        # Extraire les valeurs des sections pour cette page
        expert_text = expert_prompt if expert_mode_enabled else ""
        page_values = extract_section_values(img_path, section_titles, model, processor, config, expert_text)
        
        # Ajouter le numéro de page à chaque résultat
        for value in page_values:
            value["page"] = page_num
        
        return {
            "success": True,
            "page_num": page_num,
            "doc_name": doc_name,
            "doc_index": doc_index,
            "values": page_values
        }
        
    except Exception as e:
        print(f"Erreur lors de l'extraction de la page {page_num} de {doc_name}: {str(e)}")
        return {
            "success": False,
            "page_num": page_num,
            "doc_name": doc_name,
            "doc_index": doc_index,
            "error": str(e),
            "values": []
        }


# Fonction pour traiter plusieurs documents via l'interface Gradio
def process_multiple_documents(files, selected_sections, expert_prompt="", page_include_filter="", page_exclude_filter=""):
    """
    Extrait les valeurs des sections sélectionnées à partir d'un ou plusieurs documents.
    Le mode expert est activé automatiquement si des champs expert sont remplis.
    
    Args:
        files (list): Liste des fichiers (PDF ou images) à traiter
        selected_sections (list): Liste des sections dont les valeurs doivent être extraites
        expert_prompt (str): Instructions supplémentaires pour l'extraction (active automatiquement le mode expert)
        page_include_filter (str): Description des pages à inclure (active automatiquement le mode expert)
        page_exclude_filter (str): Description des pages à exclure (active automatiquement le mode expert)
        
    Returns:
        tuple: Un tuple contenant (1) un résumé textuel des extractions réalisées 
              et (2) le chemin vers le fichier JSON contenant les résultats détaillés
    """    
    # Démarrer le chronomètre pour mesurer le temps total
    import time
    start_time = time.time()
    print(f"🚀 DÉBUT DE L'EXTRACTION - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Détection automatique du mode expert basée sur le contenu des champs
    expert_mode_enabled = bool(
        expert_prompt.strip() or 
        page_include_filter.strip() or 
        page_exclude_filter.strip()
    )
    
    if expert_mode_enabled:
        print("🔧 MODE EXPERT ACTIVÉ AUTOMATIQUEMENT")
        if expert_prompt.strip():
            print(f"   • Instructions personnalisées: Oui ({len(expert_prompt.strip())} caractères)")
        if page_include_filter.strip():
            print(f"   • Filtre d'inclusion: {page_include_filter.strip()[:50]}...")
        if page_exclude_filter.strip():
            print(f"   • Filtre d'exclusion: {page_exclude_filter.strip()[:50]}...")
    else:
        print("📄 MODE STANDARD (aucun paramètre expert défini)")
    
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
        
        # Utiliser la version unifiée pour tous les modes
        return process_documents_unified(files, section_titles, expert_prompt, page_include_filter, page_exclude_filter, expert_mode_enabled, model, processor, config, current_mode, temp_img_paths, start_time)
    except Exception as e:
        # Calculer le temps même en cas d'erreur
        end_time = time.time()
        total_time = end_time - start_time
        print(f"❌ ERREUR LORS DE L'EXTRACTION - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  TEMPS AVANT ERREUR: {total_time:.2f} secondes ({total_time/60:.1f} minutes)")
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
def direct_api_config_update(server, model, api_key, enabled, pool_size):
    """Fonction qui met à jour la configuration API et force la mise à jour de l'interface"""
    result = update_api_config(server, model, api_key, enabled, pool_size)
    # Renvoyer aussi l'état actuel du mode pour l'affichage
    return result, display_current_mode()

# Exposer les fonctions MCP pour les assistants IA
analyze_document_for_mcp = analyze_document
extract_values_from_document = extract_values

# Interface Gradio
with gr.Blocks(title="Magic Document Scanner", theme=gr.themes.Glass()) as app:
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
                api_pool_size = gr.Slider(
                    label="Taille du pool (parallélisation)", 
                    minimum=1, 
                    maximum=20, 
                    step=1,
                    value=global_state["api_config"]["pool_size"],
                    info="Nombre de requêtes simultanées (1-20)"
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
            with gr.Row():
                page_include_filter = gr.Textbox(
                    label="Pages à inclure (description)",
                    placeholder="Ex: pages contenant des factures, documents avec en-tête officiel, formulaires remplis...",
                    lines=2
                )
                page_exclude_filter = gr.Textbox(
                    label="Pages à exclure (description)",
                    placeholder="Ex: pages vides, couvertures, pages de garde, annexes...",
                    lines=2
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
        inputs=[api_server, api_model, api_key, api_enabled, api_pool_size],
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
        inputs=[files_input, sections_output, expert_prompt, page_include_filter, page_exclude_filter],
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

def main():
    """
    Point d'entrée principal de l'application Gradio.
    
    Lance l'interface web avec le serveur MCP activé.
    """
    app.launch(
        mcp_server=True,  # Activer le serveur MCP
        share=False  # Pas de tunnel public par défaut
    )

# Lancer l'application si exécutée directement
if __name__ == "__main__":
    main()