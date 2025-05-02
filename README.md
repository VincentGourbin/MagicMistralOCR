---
title: MagicMistralOCR
app_file: app.py
sdk: gradio
sdk_version: 5.22.0
---

# Magic Document Scanner

Une application intelligente pour extraire des informations de vos documents en utilisant la puissance des modèles de vision-langage (VLM).

## Fonctionnalités

- **Détection automatique de sections** : Analyse des documents pour identifier automatiquement les titres, champs et sections
- **Saisie manuelle de sections** : Ajout de sections personnalisées pour une extraction précise
- **Traitement par lots** : Analyse de plusieurs documents du même type en une seule fois
- **Format JSON structuré** : Résultats organisés et faciles à traiter pour l'intégration à d'autres systèmes
- **Mode expert** : Personnalisez les instructions d'extraction pour des cas d'utilisation spécifiques
- **Compatibilité multiple** : 
  - Mode MLX optimisé pour Mac avec Apple Silicon
  - Mode API pour utiliser des services externes comme Mistral AI ou OpenAI
- **Serveur MCP** : Permet aux assistants IA (comme Claude Desktop) d'utiliser directement les fonctionnalités d'extraction

## Comment utiliser

### 1. Configurer l'API externe (optionnel)
- Cliquez sur "Configuration API externe" pour ouvrir le panneau
- Entrez l'URL du serveur API (format: https://api.mistral.ai/v1/chat/completions)
- Spécifiez le modèle à utiliser (ex: mistral-small-latest)
- Saisissez votre clé API et activez l'option
- Cliquez sur "Sauvegarder la configuration API"

### 2. Configurer les sections
- Téléchargez un document modèle et cliquez sur "Magic Scan" pour détecter automatiquement les sections.
- OU ajoutez manuellement des sections en les saisissant (une par ligne) et en cliquant sur "Ajouter ces sections".
- Cochez les sections que vous souhaitez extraire.

### 3. Extraire les valeurs
- Téléchargez un ou plusieurs documents du même type.
- Pour une extraction avancée, utilisez le mode expert pour personnaliser les instructions d'extraction.
- Cliquez sur "Extraire les valeurs" pour obtenir les informations des sections sélectionnées.
- Les résultats sont disponibles au format JSON.

### 4. Utiliser Magic Document Scanner avec un assistant IA (MCP)
- Lancez Magic Document Scanner avec l'option MCP activée (déjà configuré par défaut)
- Configurez votre client MCP (comme Claude Desktop) en ajoutant cette URL:
  ```json
  {
    "mcpServers": {
      "magic-scanner": {
        "url": "http://localhost:7860/gradio_api/mcp/sse"
      }
    }
  }