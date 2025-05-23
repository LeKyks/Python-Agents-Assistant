"""
Point d'entrée principal pour l'API FastAPI
"""
import logging
import os
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from config import settings
from llm.llm_connector import OllamaConnector
from llm.groq_connector import GroqConnector
from orchestrator import Orchestrator
from agents.readme_generator import ReadmeGenerator
from agents.code_assistant import CodeAssistant
from agents.debug_assistant import DebugAssistant
from agents.rag_agent import RAGAgent

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialisation de l'API
app = FastAPI(
    title="Agents Python Assistant",
    description="API pour l'assistance au développement Python via des agents spécialisés",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialisation des connecteurs LLM
ollama_connector = OllamaConnector(model_name=settings.OLLAMA_MODEL, base_url=settings.OLLAMA_URL)
groq_connector = None

if settings.GROQ_API_KEY:
    groq_connector = GroqConnector(api_key=settings.GROQ_API_KEY)
    logger.info("Groq connector initialized")
else:
    logger.warning("No Groq API key found. Groq-based features will be limited.")

# Initialisation de l'orchestrateur et des agents
orchestrator = Orchestrator()
readme_generator = ReadmeGenerator(ollama_connector)
code_assistant = CodeAssistant(ollama_connector)
debug_assistant = DebugAssistant(ollama_connector)

# Initialiser l'agent RAG avec le connecteur Groq si disponible
rag_agent = RAGAgent(groq_connector, settings.GROQ_API_KEY)

# Enregistrement des agents
orchestrator.register_agent("readme", readme_generator)
orchestrator.register_agent("code", code_assistant)
orchestrator.register_agent("debug", debug_assistant)
orchestrator.register_agent("rag", rag_agent)

# Modèles Pydantic pour la validation des données

class ReadmeRequest(BaseModel):
    project_name: str
    project_description: str
    technologies: List[str] = []
    code_snippets: List[str] = []
    include_sections: List[str] = []

class CodeRequest(BaseModel):
    code: str
    task_type: str = "correction"  # correction, optimisation, refactoring, pep8, debug
    requirements: List[str] = []
    context: str = ""

class DebugRequest(BaseModel):
    code: str
    error_message: Optional[str] = None

class RAGRequest(BaseModel):
    query: str
    index_path: Optional[str] = None
    operation: Optional[str] = "query"  # query, process, save, load

class AgentInfoResponse(BaseModel):
    id: str
    name: str
    description: str

# Endpoints

@app.get("/")
async def root():
    """Endpoint racine avec informations sur l'API"""
    return {
        "message": "Bienvenue sur l'API Agents Python Assistant",
        "version": "1.0.0",
        "docs": "/docs",
        "agents": orchestrator.get_registered_agents()
    }

@app.get("/agents", response_model=List[AgentInfoResponse])
async def list_agents():
    """Liste tous les agents disponibles"""
    return orchestrator.get_registered_agents()

@app.post("/readme/generate")
async def generate_readme(request: ReadmeRequest, background_tasks: BackgroundTasks):
    """Génère un README complet en Markdown pour un projet"""
    try:
        result = await orchestrator.process_task("readme", request.dict())
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["message"])
        
        return result["result"]
    
    except Exception as e:
        logger.error(f"Error in generate_readme: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/code/improve")
async def improve_code(request: CodeRequest):
    """Améliore le code Python selon le type de tâche demandé"""
    try:
        result = await orchestrator.process_task("code", request.dict())
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["message"])
        
        return result["result"]
    
    except Exception as e:
        logger.error(f"Error in improve_code: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/code/debug")
async def debug_code(request: DebugRequest):
    """Génère un rapport de debug pour un code Python"""
    try:
        result = await code_assistant.generate_debug_report(
            request.code, 
            request.error_message
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["message"])
        
        return result
    
    except Exception as e:
        logger.error(f"Error in debug_code: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/debug/analyze")
async def analyze_debug(request: DebugRequest):
    """Analyse un code et génère un rapport de debug détaillé"""
    try:
        result = await orchestrator.process_task("debug", request.dict())
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["message"])
        
        return result["result"]
    
    except Exception as e:
        logger.error(f"Error in analyze_debug: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/query")
async def rag_query(request: RAGRequest):
    """Répond à une question en utilisant un document traité par RAG"""
    try:
        result = await orchestrator.process_task("rag", request.dict())
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["message"])
        
        return result
    
    except Exception as e:
        logger.error(f"Error in rag_query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/process")
async def process_document(file: UploadFile = File(...)):
    """Traite un document pour le RAG"""
    try:
        # Sauvegarder temporairement le fichier
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
        
        # Traiter le document
        result = await orchestrator.process_task("rag", {
            "operation": "process",
            "file_path": file_path
        })
        
        # Nettoyer le fichier temporaire
        os.remove(file_path)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["message"])
        
        return result
    
    except Exception as e:
        logger.error(f"Error in process_document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Vérifie l'état de santé de l'API et des connecteurs LLM"""
    ollama_status = await ollama_connector.check_status()
    groq_status = groq_connector and await groq_connector.check_status()
    
    return {
        "status": "healthy" if ollama_status or groq_status else "degraded",
        "llm": {
            "ollama": {
                "model": settings.OLLAMA_MODEL,
                "status": "available" if ollama_status else "unavailable"
            },
            "groq": {
                "status": "available" if groq_status else "unavailable"
            }
        },
        "agents": orchestrator.get_registered_agents()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
