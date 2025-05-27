"""
Module d'initialisation du paquet Magic Document Scanner
"""

__version__ = "1.0.0"
__author__ = "Magic Document Scanner Team"

# Initialisation du répertoire des modules
from config import refresh_api_status, show_runtime_info
from utils import cleanup_temp_files, free_memory
from model_handler import load_model
from api_client import update_api_config
from image_processor import convert_pdf_to_images
from data_extractor import extract_sections_from_image, extract_section_values
from mcp_functions import analyze_document, extract_values