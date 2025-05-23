# c:\Users\pierr\OneDrive - Ynov\COURS\M2\NLP\Projet fil rouge\agents\base_agent.py
"""
Module définissant la classe de base pour tous les agents
"""
import abc
import logging
from typing import Dict, Any

from llm.llm_connector import LLMConnector

logger = logging.getLogger(__name__)

class BaseAgent(abc.ABC):
    """
    Agent de base définissant l'interface commune pour tous les agents spécialisés
    """
    
    def __init__(self, llm_connector: LLMConnector):
        """
        Initialise l'agent avec un connecteur LLM
        
        Args:
            llm_connector: Connecteur vers le modèle de langage à utiliser
        """
        self.llm_connector = llm_connector
        self.name = "BaseAgent"
        self.description = "Agent générique"
    
    @abc.abstractmethod
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Traite une requête spécifique à l'agent
        
        Args:
            data: Données nécessaires pour la tâche
            
        Returns:
            Dictionnaire contenant le résultat du traitement et des métadonnées
        """
        pass
    
    def check_status(self) -> bool:
        """
        Vérifie si l'agent est opérationnel (vérification du modèle LLM)
        
        Returns:
            True si l'agent est prêt à traiter des requêtes, False sinon
        """
        return self.llm_connector.check_status()
    
    def get_info(self) -> Dict[str, str]:
        """
        Renvoie les informations de base sur l'agent
        
        Returns:
            Dictionnaire contenant le nom et la description de l'agent
        """
        return {
            "name": self.name,
            "description": self.description
        }