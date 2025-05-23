# Python Agents Assistant

Un système d'assistants IA spécialisés pour aider les développeurs Python.

## Fonctionnalités principales

- **Génération de README**: Création automatique de documentation de projet
- **Assistance au code**: Amélioration, optimisation et refactoring de code Python
- **Débogage automatisé**: Analyse et correction des erreurs dans le code
- **RAG (Retrieval-Augmented Generation)**: Réponses à des questions sur des documents
- **Traitement de documents**: Extraction et analyse de contenu à partir de PDF

## Installation

1. Cloner le dépôt:
```bash
git clone https://github.com/yourusername/python-agents-assistant.git
cd python-agents-assistant
```

2. Installer les dépendances:
```bash
pip install -r requirements.txt
```

3. Configurer les variables d'environnement:
```bash
# Pour Ollama (modèle local)
export OLLAMA_MODEL=mistral  # ou autre modèle disponible localement
export OLLAMA_URL=http://localhost:11434

# Pour Groq (API externe)
export GROQ_API_KEY=votre_clé_api
```

## Utilisation

### Lancer l'API

```bash
uvicorn main:app --reload
```

L'API sera disponible à l'adresse http://localhost:8000, avec la documentation Swagger à http://localhost:8000/docs.

### Lancer l'interface utilisateur

```bash
python -m utils.ui
```

## Architecture

Le projet est structuré autour d'un orchestrateur qui coordonne différents agents spécialisés:

- `orchestrator.py`: Gère les agents et leurs interactions
- `agents/`: Contient les différents agents spécialisés
- `llm/`: Connecteurs pour les modèles de langage
- `utils/`: Utilitaires divers dont le traitement de documents

## Technologies utilisées

- FastAPI pour l'API REST
- Docling pour le traitement de documents
- LangChain pour le RAG et les chaînes de traitement
- Ollama pour l'exécution locale de modèles
- Groq pour l'accès à des modèles performants via API
- Gradio pour l'interface utilisateur

## Licence

MIT
