# c:\Users\pierr\OneDrive - Ynov\COURS\M2\NLP\Projet fil rouge\llm\llm_connector.py
"""
Module pour la connexion avec les différents modèles de langage (LLM)
"""
import logging
import os
import asyncio
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import httpx

logger = logging.getLogger(__name__)

class LLMConnector(ABC):
    """Classe abstraite définissant l'interface pour tous les connecteurs LLM"""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Génère une réponse à partir d'un prompt
        
        Args:
            prompt: Texte d'invite pour le modèle
            **kwargs: Paramètres supplémentaires spécifiques au modèle
            
        Returns:
            Texte généré par le modèle
        """
        pass
    
    @abstractmethod
    def check_status(self) -> bool:
        """
        Vérifie si le modèle est disponible et fonctionnel
        
        Returns:
            True si le modèle est disponible, False sinon
        """
        pass


class OllamaConnector(LLMConnector):
    """Connecteur pour les modèles de langage via Ollama"""
    
    def __init__(self, model_name: str = "mistral", base_url: str = "http://localhost:11434"):
        """
        Initialise le connecteur Ollama
        
        Args:
            model_name: Nom du modèle à utiliser (par défaut: "mistral")
            base_url: URL de base pour l'API Ollama
        """
        self.model_name = model_name
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)  # Timeout augmenté pour les grandes générations
        logger.info(f"Initialized OllamaConnector with model: {model_name}")
    
    async def generate(
        self, 
        prompt: str, 
        system_message: str = None,
        temperature: float = 0.7, 
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """
        Génère une réponse via Ollama
        
        Args:
            prompt: Texte d'invite pour le modèle
            system_message: Message système pour guider le comportement du modèle
            temperature: Température pour contrôler la créativité (0.0-1.0)
            max_tokens: Nombre maximum de tokens à générer
            **kwargs: Paramètres supplémentaires
            
        Returns:
            Texte généré par le modèle
        """
        try:
            logger.debug(f"Generating with Ollama model {self.model_name}, prompt length: {len(prompt)}")
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            if system_message:
                payload["system"] = system_message
            
            for key, value in kwargs.items():
                if key not in ["model", "prompt", "system"]:
                    if "options" not in payload:
                        payload["options"] = {}
                    payload["options"][key] = value
            
            response = await self.client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            
            result = response.json()
            
            if "response" in result:
                return result["response"]
            else:
                logger.error(f"Unexpected response format from Ollama: {result}")
                return ""
            
        except Exception as e:
            logger.error(f"Error generating with Ollama: {str(e)}")
            raise
    
    async def check_status(self) -> bool:
        """
        Vérifie si Ollama est disponible et si le modèle est chargé
        
        Returns:
            True si le modèle est disponible, False sinon
        """
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            
            models = response.json().get("models", [])
            available_models = [model["name"] for model in models]
            
            if self.model_name in available_models:
                logger.info(f"Model {self.model_name} is available on Ollama")
                return True
            else:
                logger.warning(f"Model {self.model_name} not found in available models: {available_models}")
                return False
                
        except Exception as e:
            logger.error(f"Error checking Ollama status: {str(e)}")
            return False