"""
Agent spécialisé dans la génération de fichiers README pour les projets
"""
import logging
from typing import Dict, Any, List

from agents.base_agent import BaseAgent
from llm.llm_connector import LLMConnector

logger = logging.getLogger(__name__)

class ReadmeGenerator(BaseAgent):
    """
    Agent spécialisé dans la génération de fichiers README complets et bien structurés
    pour les projets de développement
    """
    
    def __init__(self, llm_connector: LLMConnector):
        """
        Initialise le générateur de README
        
        Args:
            llm_connector: Connecteur vers le modèle de langage à utiliser
        """
        super().__init__(llm_connector)
        self.name = "ReadmeGenerator"
        self.description = "Agent spécialisé dans la génération de README pour les projets"

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Génère un README complet pour un projet
        
        Args:
            data: Dictionnaire contenant:
                - project_name: Nom du projet
                - project_description: Description du projet
                - code_snippets: Extraits de code représentatifs (optionnel)
                - technologies: Liste des technologies utilisées (optionnel)
                - include_sections: Sections à inclure dans le README
                
        Returns:
            Dictionnaire contenant:
                - content: Contenu complet du README au format Markdown
                - success: Booléen indiquant si la génération s'est bien déroulée
                - message: Message d'information ou d'erreur
        """
        try:
            project_name = data.get('project_name', '')
            project_description = data.get('project_description', '')
            code_snippets = data.get('code_snippets', [])
            technologies = data.get('technologies', [])
            include_sections = data.get('include_sections', [
                'Introduction', 'Installation', 'Utilisation', 'Fonctionnalités', 
                'Technologies', 'Structure du projet', 'Contribution', 'Licence'
            ])
            
            logger.info(f"Generating README for project: {project_name}")
            
            # Construction du prompt pour le modèle
            prompt = self._build_readme_prompt(
                project_name, 
                project_description, 
                code_snippets,
                technologies,
                include_sections
            )
            
            # Appel au LLM
            result = await self.llm_connector.generate(
                prompt,
                max_tokens=3000,  # Augmenté pour des READMEs plus complets
                temperature=0.7,  # Plus de liberté créative tout en gardant de la structure
                system_message="Tu es un expert en documentation technique qui excelle dans la création de README professionnels."
            )
            
            return {
                "content": result,
                "success": True,
                "message": f"README généré avec succès pour le projet {project_name}"
            }
        except Exception as e:
            logger.error(f"Error generating README: {str(e)}")
            return {
                "content": "",
                "success": False,
                "message": f"Erreur lors de la génération du README: {str(e)}"
            }

    def _build_readme_prompt(
        self, 
        project_name: str, 
        project_description: str,
        code_snippets: List[str],
        technologies: List[str],
        include_sections: List[str]
    ) -> str:
        """
        Construit un prompt pour la génération du README
        """
        tech_str = "\n".join([f"- {tech}" for tech in technologies]) if technologies else "Non spécifié"
        
        code_str = ""
        if code_snippets:
            for i, snippet in enumerate(code_snippets):
                code_str += f"\nExtrait {i+1}:\n```\n{snippet}\n```\n"
        
        sections_str = ", ".join(include_sections)
        
        return f"""Tu es un expert dans la création de documentation technique, spécialisé dans l'élaboration de README de qualité pour les projets de développement.
        
Je souhaite que tu génères un README complet au format Markdown pour le projet suivant :

## Informations sur le projet
- Nom du projet: {project_name}
- Description: {project_description}
- Technologies utilisées: 
{tech_str}

{f"## Extraits de code représentatifs du projet: {code_str}" if code_snippets else ""}

## Sections à inclure dans le README
{sections_str}

Génère un README professionnel, bien structuré et détaillé au format Markdown. 
Le README doit inclure:
1. Un en-tête attrayant avec badges pertinents
2. Une description claire et concise du projet
3. Des instructions d'installation précises et étape par étape
4. Des exemples d'utilisation avec du code bien formaté
5. Une documentation des fonctionnalités principales
6. La structure du projet si appropriée
7. Des informations sur la contribution au projet
8. Des liens vers les ressources connexes

Pour chaque section demandée, assure-toi que le contenu soit pertinent et basé sur les informations fournies.
Si des informations manquent pour certaines sections, propose un contenu générique mais utile qui pourra être personnalisé ultérieurement.

Réponds uniquement avec le contenu Markdown du README, sans commentaire supplémentaire.
"""
