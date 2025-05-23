"""
Agent spécialisé dans la génération de rapports de debug contextuels
"""
import logging
from typing import Dict, Any

from agents.base_agent import BaseAgent
from llm.llm_connector import LLMConnector

logger = logging.getLogger(__name__)

class DebugAssistant(BaseAgent):
    """
    Agent spécialisé dans la génération de rapports de debug avancés
    pour aider les développeurs à résoudre des problèmes complexes
    """
    
    def __init__(self, llm_connector: LLMConnector):
        """
        Initialise l'assistant de debug
        
        Args:
            llm_connector: Connecteur vers le modèle de langage à utiliser
        """
        super().__init__(llm_connector)
        self.name = "DebugAssistant"
        self.description = "Agent spécialisé dans l'analyse et le débogage de code Python"
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Génère un rapport de debug détaillé pour un code problématique
        
        Args:
            data: Dictionnaire contenant:
                - code: Le code problématique
                - error_message: Message d'erreur (optionnel)
                - context: Contexte d'exécution (optionnel)
                
        Returns:
            Dictionnaire contenant le rapport de debug et des métadonnées
        """
        try:
            code = data.get('code', '')
            error_message = data.get('error_message', '')
            context = data.get('context', '')
            
            if not code:
                return {
                    "debug_report": "",
                    "success": False,
                    "message": "Aucun code fourni à analyser."
                }
            
            logger.info(f"Generating debug report for Python code")
            
            # Construction du prompt
            prompt = self._build_debug_prompt(code, error_message, context)
            
            # Appel au LLM
            result = await self.llm_connector.generate(
                prompt,
                max_tokens=2048,
                temperature=0.2,  # Température basse pour des analyses précises
                system_message="Tu es un expert en débogage de code Python avec une expérience approfondie dans l'analyse et la résolution de problèmes complexes."
            )
            
            return {
                "debug_report": result,
                "success": True,
                "message": "Rapport de debug généré avec succès"
            }
            
        except Exception as e:
            logger.error(f"Error generating debug report: {str(e)}")
            return {
                "debug_report": "",
                "success": False,
                "message": f"Erreur lors de la génération du rapport de debug: {str(e)}"
            }

    def _build_debug_prompt(self, code: str, error_message: str, context: str) -> str:
        """
        Construit un prompt pour la génération du rapport de debug
        """
        error_context = f"\nMessage d'erreur rapporté:\n```\n{error_message}\n```" if error_message else ""
        execution_context = f"\nContexte d'exécution:\n{context}" if context else ""
        
        return f"""Tu es un expert en débogage de code Python avec une expérience approfondie dans l'analyse et la résolution de problèmes complexes.

Je te présente un code Python qui pose problème et nécessite une analyse approfondie :
```
{code}
```
{error_context}
{execution_context}
"""

