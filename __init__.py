"""
Magic Document Scanner

Une application intelligente de traitement de documents avec OCR et extraction de données
utilisant des modèles de vision-langage (VLM) en local ou via API.
"""

__version__ = "1.0.0"
__author__ = "Magic Document Scanner Team"
__description__ = "Application intelligente d'analyse documentaire avec IA"

# Points d'entrée principaux
from src.core.config import refresh_api_status, show_runtime_info
from src.utils.utils import cleanup_temp_files, free_memory
from src.core.model_handler import load_model
from src.core.api_client import update_api_config
from src.utils.image_processor import convert_pdf_to_images
from src.core.data_extractor import extract_sections_from_image, extract_section_values
from src.core.mcp_functions import analyze_document, extract_values

__all__ = [
    "refresh_api_status",
    "show_runtime_info", 
    "cleanup_temp_files",
    "free_memory",
    "load_model",
    "update_api_config",
    "convert_pdf_to_images",
    "extract_sections_from_image",
    "extract_section_values",
    "analyze_document",
    "extract_values"
]