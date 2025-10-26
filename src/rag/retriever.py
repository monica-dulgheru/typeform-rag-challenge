"""
RAG retrieval module for Typeform RAG prototype.

This module handles query embedding and retrieval from Pinecone vector database.
"""

import time
from typing import List, Dict, Any, Optional
import logging

# Google Cloud imports
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel

# Pinecone imports
from pinecone import Pinecone

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import config and utils
import sys
from config import (
    GCP_PROJECT_ID, get_pinecone_api_key
)
from src.utils import log_retrieval_trace

# =============================================================================
# RETRIEVAL FUNCTIONS
# =============================================================================

class RAGRetriever:
    """
    RAG retriever class for handling query embedding and vector search.
    """
    
    def __init__(self, run_id: Optional[str] = None, embedding_model: str = None, pinecone_index_name: str = None):
        """
        Initialize the RAG retriever.
        
        Args:
            run_id: Optional run ID for logging
            embedding_model: Model name for creating embeddings (required)
            pinecone_index_name: Pinecone index name (required)
        """
        if embedding_model is None:
            raise ValueError("embedding_model parameter is required")
        if pinecone_index_name is None:
            raise ValueError("pinecone_index_name parameter is required")
            
        self.run_id = run_id
        self.embedding_model_name = embedding_model
        self.pinecone_index_name = pinecone_index_name
        self.embedding_model = None
        self.pinecone_client = None
        self.index = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize embedding model and Pinecone client."""
        try:
            # Initialize Vertex AI
            aiplatform.init(
                project=GCP_PROJECT_ID,
                location="us-central1"
            )
            
            # Initialize embedding model
            self.embedding_model = TextEmbeddingModel.from_pretrained(self.embedding_model_name)
            logger.info(f"Initialized embedding model: {self.embedding_model_name}")
            
            # Initialize Pinecone with new API
            api_key = get_pinecone_api_key()
            self.pc = Pinecone(api_key=api_key)
            self.index = self.pc.Index(self.pinecone_index_name)
            logger.info(f"Initialized Pinecone index: {self.pinecone_index_name}")
            
        except Exception as e:
            logger.error(f"Error initializing RAG retriever components: {e}")
            raise
    
    def embed_query(self, query: str) -> List[float]:
        """
        Create embedding for a query.
        
        Args:
            query: User query string
            
        Returns:
            list: Query embedding vector
        """
        try:
            embeddings = self.embedding_model.get_embeddings([query])
            return embeddings[0].values
        except Exception as e:
            logger.error(f"Error creating query embedding: {e}")
            raise
    
    def retrieve_context(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: User query string
            top_k: Number of chunks to retrieve
            
        Returns:
            list: List of retrieved chunks with metadata
        """
        try:
            start_time = time.time()
            
            # Create query embedding
            query_embedding = self.embed_query(query)
            logger.info(f"Query embedding created - dimension: {len(query_embedding)}, first 3 values: {query_embedding[:3]}")
            
            # Log index stats before query to verify vectors exist
            try:
                stats = self.index.describe_index_stats()
                logger.info(f"Index stats before query: {stats}")
            except Exception as e:
                logger.warning(f"Could not get index stats before query: {e}")
            
            # Search Pinecone
            logger.info(f"Searching Pinecone with top_k={top_k}, include_metadata=True")
            search_results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Log raw search results for debugging
            logger.info(f"Raw search results: {search_results}")
            logger.info(f"Number of matches: {len(search_results.matches) if hasattr(search_results, 'matches') else 'No matches attribute'}")
            
            # Process results
            retrieved_chunks = []
            if hasattr(search_results, 'matches') and search_results.matches:
                logger.info(f"Processing {len(search_results.matches)} matches")
                for i, match in enumerate(search_results.matches):
                    logger.info(f"Match {i+1}: ID={match.id}, score={match.score}, metadata keys={list(match.metadata.keys()) if match.metadata else 'None'}")
                    chunk_data = {
                        'chunk_id': match.id,
                        'chunk_text': match.metadata.get('chunk_text', ''),
                        'parent_document_id': match.metadata.get('parent_document_id', ''),
                        'chunk_position_index': match.metadata.get('chunk_position_index', 0),
                        'doc_title': match.metadata.get('doc_title', ''),
                        'language': match.metadata.get('language', 'en'),
                        'score': match.score
                    }
                    retrieved_chunks.append(chunk_data)
            else:
                logger.warning("No matches found in search results or matches attribute missing")
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Log retrieval trace
            if self.run_id:
                log_retrieval_trace(
                    self.run_id, 
                    query, 
                    retrieved_chunks, 
                    latency_ms
                )
            
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks for query in {latency_ms:.2f}ms")
            return retrieved_chunks
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            raise
    



