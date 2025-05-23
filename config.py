# c:\Users\pierr\OneDrive - Ynov\COURS\M2\NLP\Projet fil rouge\config.py
"""
Configuration de l'application
"""
from pydantic_settings import BaseSettings
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """
    Configuration de l'application utilisant Pydantic pour la validation
    """
    # Paramètres du serveur
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    
    # Paramètres du LLM local
    LLM_TYPE: str = os.getenv("LLM_TYPE", "local")  # "local" ou "api"
    LOCAL_MODEL_PATH: str = os.getenv("LOCAL_MODEL_PATH", "./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf")
    MODEL_TYPE: str = os.getenv("MODEL_TYPE", "mistral")  # "mistral", "llama", etc.
    
    # Paramètres pour Ollama
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral")
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    
    # Paramètres pour API externes
    USE_EXTERNAL_API: bool = os.getenv("USE_EXTERNAL_API", "False").lower() == "true"
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    
    # Chemins de fichiers et dossiers
    BASE_DIR: Path = Path(__file__).resolve().parent
    LOGS_DIR: Path = BASE_DIR / "logs"
    
    # Paramètres des agents
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2048"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Instance unique des paramètres à utiliser dans l'application
settings = Settings()