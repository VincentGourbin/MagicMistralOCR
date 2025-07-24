import os
import pdf2image
from PIL import Image
from typing import List, Tuple, Dict

from core.config import TEMP_DIR, global_state

# Configuration pour les images très larges
Image.MAX_IMAGE_PIXELS = 300000000

# Fonction pour convertir un PDF en images
def convert_pdf_to_images(file_path: str, page_limit: int = 5, dpi: int = 300, doc_index: int = 0) -> List[str]:
    """
    Convertit un PDF en une liste d'images.
    
    Args:
        file_path (str): Chemin vers le fichier PDF
        page_limit (int, optional): Nombre maximum de pages à convertir
        dpi (int, optional): Résolution des images générées
        doc_index (int, optional): Indice du document pour l'état global
        
    Returns:
        List[str]: Liste des chemins vers les images générées
    """
    try:
        # Convertir le PDF en images
        images = pdf2image.convert_from_path(file_path, dpi=dpi)
        
        # Limiter le nombre de pages traitées
        images = images[:page_limit]
        
        # Sauvegarder les images temporairement
        img_paths = []
        for i, img in enumerate(images):
            # Créer un chemin unique pour l'image
            output_path = os.path.join(TEMP_DIR, f"pdf_page_{os.path.basename(file_path)}_{i}.png")
            img.save(output_path, "PNG", quality=100, dpi=(dpi, dpi))
            img_paths.append(output_path)
            
            # Stocker le chemin pour utilisation ultérieure
            global_state["image_paths"][(doc_index, i+1)] = output_path
        
        return img_paths
    except Exception as e:
        print(f"Erreur lors de la conversion du PDF en images: {str(e)}")
        return []

# Fonction pour extraire une page spécifique d'un PDF
def get_pdf_page_image(pdf_path: str, page_num: int = 0, dpi: int = 300) -> str:
    """
    Extrait une page spécifique d'un PDF en tant qu'image.
    
    Args:
        pdf_path (str): Chemin vers le fichier PDF
        page_num (int): Numéro de la page à extraire (0-indexé)
        dpi (int, optional): Résolution de l'image générée
        
    Returns:
        str: Chemin vers l'image générée ou None en cas d'erreur
    """
    try:
        # Convertir le PDF en images
        images = pdf2image.convert_from_path(pdf_path, dpi=dpi)
        
        # Vérifier si la page demandée existe
        if not images or page_num >= len(images):
            return None
        
        # Sauvegarder l'image temporairement
        temp_img_path = os.path.join(TEMP_DIR, f"temp_pdf_page_{page_num}.png")
        images[page_num].save(temp_img_path, "PNG", quality=100, dpi=(dpi, dpi))
        
        return temp_img_path
    except Exception as e:
        print(f"Erreur lors de l'extraction de la page du PDF: {str(e)}")
        return None

# Fonction pour télécharger une image depuis une URL
def download_image_from_url(image_url: str) -> str:
    """
    Télécharge une image depuis une URL.
    
    Args:
        image_url (str): URL de l'image à télécharger
        
    Returns:
        str: Chemin local vers l'image téléchargée ou None en cas d'erreur
    """
    try:
        import requests
        
        # Vérifier si c'est une URL
        if not image_url.startswith(("http://", "https://")):
            return image_url  # C'est déjà un chemin local
        
        # Télécharger l'image
        response = requests.get(image_url, timeout=30)
        if response.status_code != 200:
            print(f"Erreur lors du téléchargement de l'image: {response.status_code}")
            return None
        
        # Créer un chemin unique pour l'image
        import uuid
        temp_download_path = os.path.join(TEMP_DIR, f"download_{uuid.uuid4()}.jpg")
        
        # Sauvegarder l'image localement
        with open(temp_download_path, "wb") as f:
            f.write(response.content)
        
        return temp_download_path
    except Exception as e:
        print(f"Erreur lors du téléchargement de l'image: {str(e)}")
        return None