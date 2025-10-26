"""
RAG response generation module for Typeform RAG prototype.

This module handles LLM response generation using Gemini via Vertex AI
with retrieved context and proper prompt formatting.
"""

import time
from typing import List, Dict, Any, Optional
import logging

# LiteLLM for model abstraction
import litellm
from litellm import completion

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import config and utils
from prompts.system_prompts import (
    build_prompt_with_version, 
    extract_citations_from_response
)

from src.evaluation.evaluation import (
    analyze_response_quality
)

# =============================================================================
# RESPONSE GENERATION FUNCTIONS
# =============================================================================

class RAGGenerator:
    """
    RAG generator class for handling LLM response generation.
    """
    
    def __init__(self, run_id: Optional[str] = None, model: str = None, 
                 temperature: float = None, max_tokens: int = None):
        """
        Initialize the RAG generator.
        
        Args:
            run_id: Optional run ID for logging
            model: LLM model to use (required)
            temperature: Temperature for generation (required)
            max_tokens: Max tokens for response (required)
        """
        if model is None:
            raise ValueError("model parameter is required")
        if temperature is None:
            raise ValueError("temperature parameter is required")
        if max_tokens is None:
            raise ValueError("max_tokens parameter is required")
            
        self.run_id = run_id
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        logger.info(f"Initialized RAG generator with model: {self.model}")
    
    def generate_response(self, query: str, retrieved_chunks: List[Dict[str, Any]], 
                         prompt_version: str) -> Dict[str, Any]:
        """
        Generate response using LLM with retrieved context.
        
        Args:
            query: User query
            retrieved_chunks: List of retrieved chunks
            prompt_version: Prompt version to use (v1, v2, v3)
            
        Returns:
            dict: Response with answer, metadata, and citations
        """
        try:
            start_time = time.time()
            
            # Build prompt
            prompt = build_prompt_with_version(query, retrieved_chunks, prompt_version)
            
            # Generate response using LiteLLM
            response = completion(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract response text
            answer = response.choices[0].message.content.strip()
            
            # Calculate generation time
            generation_time = time.time() - start_time
            
            # Extract citations for logging
            citations = extract_citations_from_response(answer)
            
            # Prepare response data
            response_data = {
                'answer': answer,
                'query': query,
                'citations': citations,
                'generation_time_seconds': generation_time,
                'model_used': self.model,
                'prompt_version': prompt_version,
                'num_chunks_used': len(retrieved_chunks),
                'tokens_used': response.usage.total_tokens if hasattr(response, 'usage') else None
            }
            
            # Log generation
            logger.info(f"Generated response in {generation_time:.2f}s using {len(retrieved_chunks)} chunks")
            logger.info(f"Citations found: {len(citations)}")
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    




