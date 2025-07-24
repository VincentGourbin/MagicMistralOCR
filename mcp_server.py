#!/usr/bin/env python3
"""
Serveur MCP - Magic Document Scanner

Point d'entrée pour le protocole MCP (Model Context Protocol) permettant
l'intégration avec Claude Desktop et autres assistants IA.

Usage:
    python mcp_server.py

Configuration dans claude_desktop_config.json:
{
  "mcpServers": {
    "magic-document-scanner": {
      "command": "python",
      "args": ["/path/to/MagicMistralOCR/mcp_server.py"],
      "env": {
        "api_key": "votre-clé-api"
      }
    }
  }
}
"""

import sys
import os

# Ajouter le dossier src au path pour les imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.mcp_functions import analyze_document, extract_values

def main():
    """Point d'entrée principal du serveur MCP."""
    try:
        # Configuration basique du serveur MCP
        print("Magic Document Scanner MCP Server démarré")
        print("Fonctions disponibles:")
        print("- analyze_document(image_url): Analyse un document")
        print("- extract_values(image_url, sections, expert_instructions): Extrait des valeurs")
        
        # Note: Un vrai serveur MCP nécessiterait l'implémentation complète du protocole
        # Cette version est un placeholder pour la documentation
        
    except Exception as e:
        print(f"Erreur lors du démarrage du serveur MCP: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()