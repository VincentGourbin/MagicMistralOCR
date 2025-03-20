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
- **Double compatibilité** : Fonctionne à la fois sur Mac avec MLX et sur toutes les plateformes avec Hugging Face Transformers

## Comment utiliser

### 1. Configurer les sections
- Téléchargez un document modèle et cliquez sur "Magic Scan" pour détecter automatiquement les sections.
- OU ajoutez manuellement des sections en les saisissant (une par ligne) et en cliquant sur "Ajouter ces sections".
- Cochez les sections que vous souhaitez extraire.

### 2. Extraire les valeurs
- Téléchargez un ou plusieurs documents du même type.
- Cliquez sur "Extraire les valeurs" pour obtenir les informations des sections sélectionnées.
- Les résultats sont disponibles au format JSON.

## Configuration technique

L'application détecte automatiquement l'environnement d'exécution :

- **Sur Mac avec Apple Silicon** : Utilise MLX pour une exécution optimisée avec le modèle mlx-community/Mistral-Small-3.1-24B-Instruct-2503-8bit
- **Sur les autres plateformes** : Utilise Hugging Face Transformers avec le modèle mistralai/Mistral-Small-3.1-24B-Instruct-2503

## Installation locale

### Sur Mac Apple Silicon

```bash
# Installer les dépendances
pip install -r mac-requirements.txt

# Exécuter l'application
python app.py
```

### Sur d'autres plateformes

```bash
# Installer les dépendances
pip install -r requirements.txt

# Exécuter l'application
python app.py
```

## Notes importantes

- Pour les PDF, seules les 5 premières pages sont traitées pour limiter le temps d'analyse
- La qualité de l'extraction dépend de la clarté du document et de la qualité de l'image