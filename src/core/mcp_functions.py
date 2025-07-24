"""
Fonctions MCP - Magic Document Scanner

Ce module contient les fonctions exposées via le protocole MCP :
- Analyse automatique de documents
- Extraction de valeurs spécifiques
- Interface standardisée pour les assistants IA

Functions:
- analyze_document(): Analyse complète d'un document
- extract_values(): Extraction de valeurs spécifiques
"""

import os
import json
from typing import Dict, Any, List

from core.config import TEMP_DIR, global_state
from core.model_handler import load_model
from core.data_extractor import extract_sections_from_image, extract_section_values
from utils.image_processor import download_image_from_url, convert_pdf_to_images, get_pdf_page_image

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
        file_path = download_image_from_url(image_url)
        if not file_path:
            return {"error": "Impossible de télécharger ou d'accéder au fichier", "sections": []}
            
        # Charger le modèle
        model, processor, config = load_model()
        
        # Traiter le fichier
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()
        
        all_sections = []
        temp_img_paths = []
        
        if file_extension in ['.pdf']:
            # Convertir PDF en images
            temp_img_paths = convert_pdf_to_images(file_path, page_limit=5, dpi=300)
            
            # Traiter chaque image
            for i, img_path in enumerate(temp_img_paths):
                # Traiter l'image avec le modèle ou l'API
                page_sections = extract_sections_from_image(img_path, model, processor, config)
                
                # Ajouter les titres trouvés avec indication de la page
                for section in page_sections:
                    section["page"] = i + 1
                    section["doc_index"] = 0
                    all_sections.append(section)
        else:
            # Pour les images directement
            all_sections = extract_sections_from_image(file_path, model, processor, config)
            for section in all_sections:
                section["page"] = 1
                section["doc_index"] = 0
        
        return {"sections": all_sections, "total": len(all_sections)}
    except Exception as e:
        print(f"Erreur lors de l'analyse du document: {str(e)}")
        return {"error": str(e), "sections": []}
    finally:
        # Nettoyer tous les fichiers temporaires
        if 'file_path' in locals() and file_path and file_path.startswith(TEMP_DIR) and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Erreur lors du nettoyage du fichier téléchargé: {str(e)}")
        
        for path in temp_img_paths:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"Erreur lors du nettoyage de l'image temporaire: {str(e)}")

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
        file_path = download_image_from_url(image_url)
        if not file_path:
            return {"error": "Impossible de télécharger ou d'accéder au fichier", "extracted_values": []}
        
        # Vérifier si le fichier est un PDF
        _, file_extension = os.path.splitext(file_path)
        is_pdf = file_extension.lower() == '.pdf'
            
        # Charger le modèle
        model, processor, config = load_model()
        
        # Préparation de l'expert prompt
        expert_text = expert_instructions if expert_instructions else ""
        
        # Variables pour suivre les valeurs extraites et les fichiers temporaires
        all_extracted_values = []
        temp_img_paths = []
        
        if is_pdf:
            # Convertir le PDF en images
            temp_img_paths = convert_pdf_to_images(file_path, page_limit=5, dpi=300)
            
            # Traiter chaque image
            for i, img_path in enumerate(temp_img_paths):
                # Extraire les valeurs pour cette page
                page_values = extract_section_values(img_path, sections, model, processor, config, expert_text)
                
                # Ajouter le numéro de page
                for value in page_values:
                    value["page"] = i + 1
                    
                # Filtrer les valeurs avec une confiance suffisante
                filtered_values = [v for v in page_values if v.get("confidence", 0) > 0.2]
                all_extracted_values.extend(filtered_values)
        else:
            # Pour les images directement
            page_values = extract_section_values(file_path, sections, model, processor, config, expert_text)
            
            # Ajouter le numéro de page
            for value in page_values:
                value["page"] = 1
                
            all_extracted_values.extend(page_values)
        
        # Dédupliquer les valeurs en gardant celles avec la confiance la plus élevée
        sections_dict = {}
        for item in all_extracted_values:
            section = item["section"]
            confidence = item.get("confidence", 0)
            
            if section not in sections_dict or confidence > sections_dict[section].get("confidence", 0):
                sections_dict[section] = item
                
        # Convertir le dictionnaire en liste
        final_values = list(sections_dict.values())
        
        # Transformer la sortie en dictionnaire pour une meilleure lisibilité
        result = {}
        for item in final_values:
            result[item["section"]] = {
                "value": item["value"],
                "confidence": item["confidence"],
                "page": item.get("page", 1)
            }
        
        return {"extracted_values": final_values, "results": result}
    except Exception as e:
        print(f"Erreur lors de l'extraction des valeurs: {str(e)}")
        return {"error": str(e), "extracted_values": []}
    finally:
        # Nettoyer tous les fichiers temporaires
        if 'file_path' in locals() and file_path and file_path.startswith(TEMP_DIR) and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Erreur lors du nettoyage du fichier téléchargé: {str(e)}")
        
        for path in temp_img_paths:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"Erreur lors du nettoyage de l'image temporaire: {str(e)}")