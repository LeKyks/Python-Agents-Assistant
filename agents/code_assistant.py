"""
Agent spécialisé dans l'assistance et amélioration de code Python
"""
import logging
from typing import Dict, Any

from agents.base_agent import BaseAgent
from llm.llm_connector import LLMConnector

logger = logging.getLogger(__name__)

class CodeAssistant(BaseAgent):
    """
    Agent spécialisé dans l'amélioration, correction et optimisation du code Python
    """
    
    def __init__(self, llm_connector: LLMConnector):
        """
        Initialise l'assistant de code
        
        Args:
            llm_connector: Connecteur vers le modèle de langage à utiliser
        """
        super().__init__(llm_connector)
        self.name = "CodeAssistant"
        self.description = "Agent spécialisé dans l'amélioration et le débogage de code Python"
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Améliore le code Python selon le type de tâche demandé
        
        Args:
            data: Dictionnaire contenant:
                - code: Code Python à traiter
                - task_type: Type de tâche (correction, optimisation, refactoring, pep8, debug)
                - requirements: Exigences spécifiques
                - context: Contexte d'utilisation du code
                
        Returns:
            Dictionnaire contenant le code amélioré et des métadonnées
        """
        try:
            code = data.get('code', '')
            task_type = data.get('task_type', 'correction')
            requirements = data.get('requirements', [])
            context = data.get('context', '')
            
            if not code:
                return {
                    "improved_code": "",
                    "explanation": "Aucun code fourni à améliorer.",
                    "success": False,
                    "message": "Erreur: code manquant"
                }
            
            logger.info(f"Processing code improvement task: {task_type}")
            
            # Construction du prompt selon le type de tâche
            prompt = self._build_code_prompt(code, task_type, requirements, context)
            
            # Appel au LLM
            result = await self.llm_connector.generate(
                prompt,
                max_tokens=2048,
                temperature=0.2,  # Température basse pour des réponses précises et cohérentes
                system_message="Tu es un expert en Python qui excelle dans l'amélioration et l'optimisation de code."
            )
            
            # Extraction du code et des explications de la réponse
            improved_code, explanation = self._parse_llm_response(result)
            
            return {
                "improved_code": improved_code,
                "explanation": explanation,
                "success": True,
                "message": f"Code {task_type} effectué avec succès"
            }
        
        except Exception as e:
            logger.error(f"Error in code improvement: {str(e)}")
            return {
                "improved_code": "",
                "explanation": "",
                "success": False,
                "message": f"Erreur lors de l'amélioration du code: {str(e)}"
            }
    
    async def generate_debug_report(self, code: str, error_message: str = None) -> Dict[str, Any]:
        """
        Génère un rapport de debug pour un code Python
        
        Args:
            code: Code Python à déboguer
            error_message: Message d'erreur éventuel
            
        Returns:
            Dictionnaire contenant le rapport de debug et des métadonnées
        """
        try:
            if not code:
                return {
                    "debug_report": "",
                    "fixed_code": "",
                    "success": False,
                    "message": "Aucun code fourni à déboguer"
                }
            
            logger.info("Generating debug report")
            
            # Construction du prompt pour le debug
            prompt = self._build_debug_prompt(code, error_message)
            
            # Appel au LLM
            result = await self.llm_connector.generate(
                prompt,
                max_tokens=2048,
                temperature=0.3,
                system_message="Tu es un expert en débogage Python capable d'identifier et de résoudre rapidement les problèmes complexes."
            )
            
            # Parse the response to extract the debug report and fixed code
            debug_report, fixed_code = self._parse_debug_response(result)
            
            return {
                "debug_report": debug_report,
                "fixed_code": fixed_code,
                "success": True,
                "message": "Rapport de debug généré avec succès"
            }
            
        except Exception as e:
            logger.error(f"Error generating debug report: {str(e)}")
            return {
                "debug_report": "",
                "fixed_code": "",
                "success": False,
                "message": f"Erreur lors de la génération du rapport de debug: {str(e)}"
            }

    def _build_code_prompt(
        self, 
        code: str, 
        task_type: str, 
        requirements: list, 
        context: str
    ) -> str:
        """
        Construit un prompt pour l'amélioration de code
        """
        task_descriptions = {
            "correction": "Corrige les erreurs dans le code tout en préservant sa fonctionnalité originale.",
            "optimisation": "Optimise le code pour améliorer ses performances (vitesse d'exécution, utilisation de la mémoire).",
            "refactoring": "Réorganise le code pour améliorer sa lisibilité et sa maintenabilité sans changer son comportement.",
            "pep8": "Modifie le code pour respecter les conventions de style PEP 8 de Python.",
            "debug": "Identifie les problèmes potentiels dans le code et propose des solutions."
        }
        
        task_desc = task_descriptions.get(task_type, "Améliore le code Python fourni.")
        req_str = "\n".join([f"- {req}" for req in requirements]) if requirements else "Aucune exigence spécifique."
        
        return f"""En tant qu'expert Python, ta tâche est de: {task_desc}

Code Python à améliorer:
```python
{code}
```

Exigences spécifiques:
{req_str}

{"Contexte d'utilisation:\n" + context if context else ""}

Instructions:
1. Analyse attentivement le code fourni
2. {task_desc}
3. Fournis le code amélioré
4. Explique les modifications importantes que tu as apportées

Réponds en fournissant d'abord le code amélioré encadré par ```python et ```, puis une explication claire des modifications.
"""

    def _build_debug_prompt(self, code: str, error_message: str = None) -> str:
        """
        Construit un prompt pour le débogage
        """
        error_context = f"\nLe code génère l'erreur suivante:\n```\n{error_message}\n```" if error_message else ""
        
        return f"""En tant qu'expert en débogage Python, analyse ce code et identifie ses problèmes:

```python
{code}
```
{error_context}

Instructions:
1. Analyse le code et identifie tous les problèmes (erreurs de syntaxe, bugs logiques, inefficacités, etc.)
2. Explique chaque problème identifié et pourquoi il pose problème
3. Propose une solution correcte pour chaque problème
4. Fournis une version corrigée et améliorée du code

Réponds avec:
1. Un rapport de débogage détaillé expliquant les problèmes
2. Le code corrigé entre balises ```python et ```
"""

    def _parse_llm_response(self, response: str) -> tuple:
        """
        Extrait le code et l'explication d'une réponse LLM
        
        Args:
            response: Texte brut de la réponse du modèle
            
        Returns:
            Tuple contenant (code amélioré, explication)
        """
        # Recherche le code entre les marqueurs de bloc de code Python
        import re
        
        code_pattern = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL)
        code_match = code_pattern.search(response)
        
        if code_match:
            improved_code = code_match.group(1).strip()
            # L'explication est tout ce qui suit le dernier bloc de code
            explanation = response[code_match.end():].strip()
            
            # Si l'explication est vide, prendre tout ce qui précède le premier bloc de code
            if not explanation:
                explanation = response[:code_match.start()].strip()
                
            return improved_code, explanation
        
        # Fallback: s'il n'y a pas de bloc de code correctement formaté
        return response, ""
    
    def _parse_debug_response(self, response: str) -> tuple:
        """
        Extrait le rapport de debug et le code corrigé d'une réponse LLM
        """
        import re
        
        # Recherche le code entre les marqueurs de bloc de code Python
        code_pattern = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL)
        code_match = code_pattern.search(response)
        
        if code_match:
            fixed_code = code_match.group(1).strip()
            
            # Le rapport de debug est tout ce qui précède le bloc de code
            debug_report = response[:code_match.start()].strip()
            
            # Si le rapport est vide, prendre tout ce qui suit le bloc de code
            if not debug_report:
                debug_report = response[code_match.end():].strip()
                
            return debug_report, fixed_code
        
        # S'il n'y a pas de bloc de code, considérer la réponse entière comme un rapport
        return response, ""
