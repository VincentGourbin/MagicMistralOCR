#!/usr/bin/env python3
"""
Magic Document Scanner - Point d'entrée principal

Application de traitement intelligent de documents avec OCR et extraction de données
utilisant des modèles de vision-langage (VLM) en local (MLX) ou via API externe.

Usage:
    python main.py

Author: Magic Document Scanner Team
"""

import sys
import os

# Ajouter le dossier src au path Python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ui.app import main

if __name__ == "__main__":
    main()