"""
Agent for Retrieval-Augmented Generation (RAG) using Docling for document processing
"""
import logging
import os
from typing import Dict, Any, List, Optional

import langchain
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from agents.base_agent import BaseAgent
from utils.document_processor import DoclingProcessor

logger = logging.getLogger(__name__)

class RAGAgent(BaseAgent):
    """
    Agent implementing Retrieval-Augmented Generation (RAG) capabilities
    with Docling for document processing
    """
    
    def __init__(self, llm_connector=None, groq_api_key: str = None):
        """
        Initialize the RAG agent
        
        Args:
            llm_connector: LLM connector to use (optional)
            groq_api_key: API key for Groq (if not using llm_connector)
        """
        super().__init__(llm_connector)
        self.name = "RAGAgent"
        self.description = "Agent for document Q&A using Retrieval-Augmented Generation"
        
        # Setup RAG components
        self.embed_model_id = "sentence-transformers/all-MiniLM-L6-v2"
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embed_model_id)
        self.vectorstore = None
        self.doc_processor = DoclingProcessor(embed_model_id=self.embed_model_id)
        
        # Setup Groq as a fallback if no LLM connector is provided
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        
        if self.groq_api_key and not self.llm_connector:
            logger.info("Using Groq LLM for RAG agent")
            self.llm = ChatGroq(
                api_key=self.groq_api_key,
                model_name="llama3-70b-8192",
                temperature=0.7,
                max_tokens=2048
            )
        else:
            logger.info("Using provided LLM connector for RAG agent")
            self.llm = None  # Will use self.llm_connector in process method
    
    async def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process a document and build a vector index
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with processing results and status
        """
        try:
            # Process document with Docling
            result, chunks = self.doc_processor.process_document(file_path)
            
            # Convert chunks to Langchain documents
            documents = [
                Document(page_content=chunk, metadata={"source": f"chunk_{i}"})
                for i, chunk in enumerate(chunks)
            ]
            
            # Create vector index
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            
            # Extract images
            images = self.doc_processor.get_images_from_document(result)
            
            return {
                "success": True,
                "message": f"Document processed successfully: {len(documents)} chunks created",
                "document_info": {
                    "chunks": len(documents),
                    "images": len(images)
                }
            }
        
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return {
                "success": False,
                "message": f"Error processing document: {str(e)}",
                "document_info": None
            }
    
    async def save_index(self, index_path: str) -> Dict[str, Any]:
        """
        Save the current vector index
        
        Args:
            index_path: Path to save the index
            
        Returns:
            Dictionary with operation status
        """
        try:
            if not self.vectorstore:
                return {
                    "success": False,
                    "message": "No index to save. Process a document first."
                }
            
            self.vectorstore.save_local(index_path)
            return {
                "success": True,
                "message": f"Index saved successfully to {index_path}"
            }
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            return {
                "success": False,
                "message": f"Error saving index: {str(e)}"
            }
    
    async def load_index(self, index_path: str) -> Dict[str, Any]:
        """
        Load a saved vector index
        
        Args:
            index_path: Path to the index
            
        Returns:
            Dictionary with operation status
        """
        try:
            self.vectorstore = FAISS.load_local(index_path, self.embeddings)
            return {
                "success": True,
                "message": f"Index loaded successfully from {index_path}"
            }
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return {
                "success": False,
                "message": f"Error loading index: {str(e)}"
            }
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a query with the RAG system
        
        Args:
            data: Dictionary containing:
                - query: Question to answer
                - file_path: Optional path to document to process
                - index_path: Optional path to load/save index
                - operation: Optional operation ('process', 'save', 'load')
                
        Returns:
            Dictionary with response and metadata
        """
        try:
            operation = data.get('operation', 'query')
            
            # Handle different operations
            if operation == 'process' and 'file_path' in data:
                return await self.process_document(data['file_path'])
            
            if operation == 'save' and 'index_path' in data:
                return await self.save_index(data['index_path'])
                
            if operation == 'load' and 'index_path' in data:
                return await self.load_index(data['index_path'])
            
            # Handle queries
            query = data.get('query')
            if not query:
                return {
                    "success": False,
                    "message": "No query provided",
                    "answer": None
                }
                
            if not self.vectorstore:
                return {
                    "success": False,
                    "message": "No document has been processed. Process a document first.",
                    "answer": None
                }
            
            # Setup the RAG prompt
            prompt_template = """
            Tu es un assistant spécialisé en traitement de documents et NLP.
            Utilise uniquement les informations contextuelles ci-dessous pour répondre à la question.
            Si les informations ne sont pas suffisantes, dis simplement que tu ne sais pas.

            Informations contextuelles:
            {context}

            Question: {question}

            Ta réponse (sois précis et détaillé):
            """
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Use the appropriate LLM
            llm_to_use = self.llm if self.llm else self.llm_connector
            
            # Create and run RAG chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm_to_use,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt}
            )
            
            result = qa_chain.invoke({"query": query})
            
            # Extract source chunks for reference
            source_chunks = []
            if "source_documents" in result:
                for doc in result["source_documents"]:
                    source_chunks.append({
                        "source": doc.metadata.get("source", "unknown"),
                        "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    })
            
            return {
                "success": True,
                "message": "Query processed successfully",
                "answer": result["result"],
                "sources": source_chunks
            }
            
        except Exception as e:
            logger.error(f"Error processing RAG query: {str(e)}")
            return {
                "success": False,
                "message": f"Error processing query: {str(e)}",
                "answer": None
            }
