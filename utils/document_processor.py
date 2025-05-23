"""
Module for document processing using Docling
"""
import logging
import os
from typing import List, Dict, Any, Tuple

from docling.document_converter import DocumentConverter, ConversionResult
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class DoclingProcessor:
    """
    Document processor that uses Docling to extract and process documents.
    """
    
    def __init__(self, embed_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the Docling document processor
        
        Args:
            embed_model_id: ID of the embedding model to use for tokenization
        """
        self.embed_model_id = embed_model_id
        self.converter = DocumentConverter()
        
        # Initialize HuggingFace tokenizer for chunking
        self.tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(embed_model_id),
        )
        
        self.chunker = HybridChunker(tokenizer=self.tokenizer)
        logger.info(f"Initialized DoclingProcessor with model: {embed_model_id}")
    
    def process_document(self, file_path: str) -> Tuple[ConversionResult, List[str]]:
        """
        Process a document file and split it into chunks
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Tuple containing the conversion result and a list of text chunks
        """
        try:
            # Convert the document
            logger.info(f"Converting document: {file_path}")
            result = self.converter.convert(file_path)
            
            # Chunk the document
            logger.info("Chunking document...")
            chunk_iter = self.chunker.chunk(dl_doc=result.document)
            
            # Extract text from chunks
            documents = []
            for i, chunk in enumerate(chunk_iter):
                enriched_text = self.chunker.contextualize(chunk=chunk)
                documents.append(enriched_text)
                
            logger.info(f"Document processed successfully: {len(documents)} chunks created")
            return result, documents
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise
    
    def get_images_from_document(self, result: ConversionResult) -> List[Dict[str, Any]]:
        """
        Extract images from a document
        
        Args:
            result: Conversion result from docling
            
        Returns:
            List of dictionaries containing image data and metadata
        """
        images = []
        try:
            for i, picture in enumerate(result.document.pictures):
                images.append({
                    "index": i,
                    "uri": picture.image.uri,
                    "page": getattr(picture, "page_no", None),
                    "caption": getattr(picture, "caption", None)
                })
            logger.info(f"Extracted {len(images)} images from document")
            return images
        except Exception as e:
            logger.error(f"Error extracting images: {str(e)}")
            return []
