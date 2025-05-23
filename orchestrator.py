"""
Agent orchestrateur qui coordonne les autres agents spécialisés
"""
import logging
from typing import Dict, Any, List

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Agent orchestrateur qui coordonne l'exécution des agents spécialisés
    en fonction des tâches demandées
    """
    
    def __init__(self):
        """
        Initialise l'orchestrateur
        """
        self.agents: Dict[str, BaseAgent] = {}
    
    def register_agent(self, agent_id: str, agent: BaseAgent) -> None:
        """
        Enregistre un agent auprès de l'orchestrateur
        
        Args:
            agent_id: Identifiant unique de l'agent
            agent: Instance de l'agent à enregistrer
        """
        self.agents[agent_id] = agent
        logger.info(f"Agent registered: {agent_id} ({agent.name})")
    
    def unregister_agent(self, agent_id: str) -> None:
        """
        Supprime un agent de l'orchestrateur
        
        Args:
            agent_id: Identifiant unique de l'agent à supprimer
        """
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Agent unregistered: {agent_id}")
    
    def get_registered_agents(self) -> List[Dict[str, str]]:
        """
        Renvoie la liste des agents enregistrés
        
        Returns:
            Liste des agents avec leurs informations
        """
        return [
            {"id": agent_id, **agent.get_info()}
            for agent_id, agent in self.agents.items()
        ]
    
    async def process_task(self, agent_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Délègue une tâche à l'agent approprié
        
        Args:
            agent_id: Identifiant de l'agent à utiliser
            data: Données à traiter par l'agent
            
        Returns:
            Résultat du traitement par l'agent
        """
        if agent_id not in self.agents:
            logger.error(f"Agent not found: {agent_id}")
            return {
                "success": False,
                "message": f"Agent non trouvé: {agent_id}",
                "result": None
            }
        
        try:
            agent = self.agents[agent_id]
            
            if not agent.check_status():
                logger.error(f"Agent {agent_id} is not available")
                return {
                    "success": False,
                    "message": f"Agent {agent_id} non disponible",
                    "result": None
                }
            
            logger.info(f"Delegating task to agent: {agent_id}")
            result = await agent.process(data)
            
            return {
                "success": True,
                "message": f"Tâche traitée par {agent_id}",
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Error processing task with agent {agent_id}: {str(e)}")
            return {
                "success": False,
                "message": f"Erreur lors du traitement: {str(e)}",
                "result": None
            }
