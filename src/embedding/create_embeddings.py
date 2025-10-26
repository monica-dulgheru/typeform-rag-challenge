"""
Embedding creation module for Typeform RAG prototype.

This module loads chunks and creates embeddings using Google's text-embedding-005
via Vertex AI, then stores them in Pinecone vector database.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Google Cloud and Pinecone imports
from google.cloud import aiplatform
from pinecone import Pinecone, ServerlessSpec
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import config and utils
import sys
from config import (
    GCP_PROJECT_ID,
    construct_path, get_pinecone_api_key
)
from src.utils import (
    load_artifacts, save_artifacts, log_workflow_step, log_metrics
)

# =============================================================================
# EMBEDDING FUNCTIONS
# =============================================================================

def initialize_vertex_ai() -> None:
    """
    Initialize Vertex AI client.
    """
    try:
        # Initialize Vertex AI
        aiplatform.init(
            project=GCP_PROJECT_ID,
            location="us-central1"  # Vertex AI region
        )
        logger.info("Vertex AI initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Vertex AI: {e}")
        raise

def create_embeddings_batch(texts: List[str], embedding_model: str, batch_size: int = 100) -> List[List[float]]:
    """
    Create embeddings for a batch of texts using Google's text-embedding-005 with parallel processing.
    
    Args:
        texts: List of text strings to embed
        embedding_model: Model name for creating embeddings
        batch_size: Number of texts to process in each batch
        
    Returns:
        list: List of embedding vectors
    """
    try:
        from vertexai.language_models import TextEmbeddingModel
        
        # Initialize the embedding model
        model = TextEmbeddingModel.from_pretrained(embedding_model)
        
        # Pre-allocate list to maintain order
        all_embeddings = [None] * len(texts)
        
        # Create batches with their starting indices
        batches = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batches.append((i, batch_texts))
        
        logger.info(f"Processing {len(batches)} embedding batches in parallel")
        
        # Use ThreadPoolExecutor for parallel batch processing
        max_workers = min(3, len(batches))  # Use up to 3 workers to avoid rate limits
        logger.info(f"Using {max_workers} workers for parallel embedding creation")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches for processing
            future_to_batch = {
                executor.submit(model.get_embeddings, batch_texts): (batch_idx, batch_texts)
                for batch_idx, batch_texts in batches
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch_idx, batch_texts = future_to_batch[future]
                try:
                    # Create embeddings for the batch
                    embeddings = future.result()
                    
                    # Store embeddings in correct positions
                    for i, embedding in enumerate(embeddings):
                        all_embeddings[batch_idx + i] = embedding.values
                    
                    logger.info(f"Completed embedding batch starting at index {batch_idx}: {len(embeddings)} embeddings")
                    
                except Exception as e:
                    logger.error(f"Error processing batch starting at index {batch_idx}: {e}")
                    # Continue with other batches
                    continue
        
        # Filter out any None values (failed batches)
        valid_embeddings = [emb for emb in all_embeddings if emb is not None]
        
        if len(valid_embeddings) != len(texts):
            logger.warning(f"Expected {len(texts)} embeddings but got {len(valid_embeddings)}")
        
        logger.info(f"Created {len(valid_embeddings)} embeddings using parallel processing")
        return valid_embeddings
        
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        raise

def wait_for_index_ready(index, max_wait_seconds: int = 60) -> bool:
    """
    Poll index stats until vectors are confirmed indexed or timeout.
    
    Args:
        index: Pinecone index object
        max_wait_seconds: Maximum time to wait
        
    Returns:
        bool: True if index is ready, False if timeout
    """
    import time
    
    start_time = time.time()
    while time.time() - start_time < max_wait_seconds:
        try:
            stats = index.describe_index_stats()
            total_vector_count = stats.get('total_vector_count', 0)
            logger.info(f"Index readiness check: {total_vector_count} vectors indexed")
            
            if total_vector_count > 0:
                logger.info("Index is ready with vectors")
                return True
                
        except Exception as e:
            logger.warning(f"Error checking index readiness: {e}")
        
        time.sleep(2)  # Wait 2 seconds before next check
    
    logger.warning(f"Index readiness timeout after {max_wait_seconds} seconds")
    return False

def initialize_pinecone(embedding_dimension: int, pinecone_index_name: str, pinecone_environment: str):
    """
    Initialize Pinecone client and ensure index exists.
    
    Args:
        embedding_dimension: Dimension of embeddings
        pinecone_index_name: Name of the Pinecone index
        pinecone_environment: Pinecone environment/region
    
    PROTOTYPE BEHAVIOR: Deletes and recreates the index for each experiment run.
    This is done for simplicity and to work within free tier index limits.
    
    FOR FUTURE/PRODUCTION DEVELOPMENT: Index names should include the run_id
    (e.g., 'typeform-help-rag-{run_id}') to enable proper experiment versioning
    and avoid data loss between runs.
    
    Returns:
        pinecone.Index: Initialized Pinecone index
    """
    try:
        # Initialize Pinecone with new API
        api_key = get_pinecone_api_key()
        pc = Pinecone(api_key=api_key)
        
        # Check if index exists
        existing_indexes = pc.list_indexes()
        if pinecone_index_name in [idx.name for idx in existing_indexes]:
            logger.info(f"Deleting existing Pinecone index: {pinecone_index_name}")
            # Delete existing index for prototype simplicity
            # TODO: In production, use run-specific index names (e.g., index-name-{run_id})
            # to preserve experiment data and enable proper versioning
            pc.delete_index(pinecone_index_name)
            logger.info("Index deleted successfully")
        
        logger.info(f"Creating new Pinecone index: {pinecone_index_name}")
        
        # Create serverless index (free tier compatible)
        pc.create_index(
            name=pinecone_index_name,
            dimension=embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        
        # Wait for index to be ready and verify it's properly initialized
        logger.info("Waiting for index to be ready...")
        time.sleep(10)  # Wait for index creation
        
        # Get index and verify it's ready
        index = pc.Index(pinecone_index_name)
        
        # Log index stats to verify readiness
        try:
            stats = index.describe_index_stats()
            logger.info(f"Index stats after creation: {stats}")
            logger.info(f"Index dimension: {embedding_dimension}")
        except Exception as e:
            logger.warning(f"Could not get index stats immediately: {e}")
        
        return index
        
    except Exception as e:
        logger.error(f"Error initializing Pinecone: {e}")
        raise

def upsert_vectors_to_pinecone(chunks: List[Dict[str, Any]], embeddings: List[List[float]], index) -> None:
    """
    Upsert vectors and metadata to Pinecone index.
    
    Args:
        chunks: List of chunk dictionaries
        embeddings: List of embedding vectors
        index: Pinecone index
    """
    try:
        # Use the provided index
        
        # Prepare vectors for upsert
        vectors_to_upsert = []
        
        # Log first vector details for debugging
        if chunks and embeddings:
            logger.info(f"First vector details - ID: {chunks[0]['chunk_id']}, embedding dimension: {len(embeddings[0])}")
            logger.info(f"Total vectors to upsert: {len(chunks)}")
        
        for chunk, embedding in zip(chunks, embeddings):
            # Prepare metadata (Pinecone has size limits on metadata)
            metadata = {
                'chunk_id': chunk['chunk_id'],
                'chunk_text': chunk['chunk_text'][:1000],  # Limit text length in metadata
                'parent_document_id': chunk['parent_document_id'],
                'chunk_position_index': chunk['chunk_position_index'],
                'doc_title': chunk.get('doc_title', '')[:500],  # Limit title length
                'language': chunk.get('language', 'en')
            }
            
            vector_data = {
                'id': chunk['chunk_id'],
                'values': embedding,
                'metadata': metadata
            }
            
            vectors_to_upsert.append(vector_data)
        
        # Upsert in batches (increased batch size for better performance)
        batch_size = 200
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            
            logger.info(f"Upserting batch {i//batch_size + 1}/{(len(vectors_to_upsert) + batch_size - 1)//batch_size}")
            
            try:
                index.upsert(vectors=batch)
                time.sleep(0.5)  # Small delay between batches
            except Exception as e:
                logger.error(f"Error upserting batch {i//batch_size + 1}: {e}")
                continue
        
        logger.info(f"Successfully upserted {len(vectors_to_upsert)} vectors to Pinecone")
        
        # Log index stats after upsert to verify vectors are indexed
        try:
            stats = index.describe_index_stats()
            logger.info(f"Index stats after upsert: {stats}")
            
            # Wait for index to be ready with polling
            logger.info("Waiting for index to be ready for queries...")
            if wait_for_index_ready(index, max_wait_seconds=30):
                logger.info("Index confirmed ready for queries")
            else:
                logger.warning("Index may not be fully ready for queries")
                
        except Exception as e:
            logger.warning(f"Could not get index stats after upsert: {e}")
        
    except Exception as e:
        logger.error(f"Error upserting vectors to Pinecone: {e}")
        raise

def save_embedding_metadata(run_id: str, embedding_stats: Dict[str, Any], embedding_model: str, embedding_dimension: int, pinecone_index_name: str) -> None:
    """
    Save embedding metadata and statistics.
    
    Args:
        run_id: Run ID for the experiment
        embedding_stats: Statistics about the embedding process
        embedding_model: Model name for creating embeddings
        embedding_dimension: Dimension of embeddings
    """
    try:
        metadata = {
            'embedding_model': embedding_model,
            'embedding_dimension': embedding_dimension,
            'pinecone_index_name': pinecone_index_name,  # Prototype: single shared index
            'created_at': datetime.now().isoformat(),
            'stats': embedding_stats
        }
        
        save_artifacts(metadata, run_id, "embeddings_metadata")
        logger.info("Saved embedding metadata")
        
    except Exception as e:
        logger.error(f"Error saving embedding metadata: {e}")
        raise

# =============================================================================
# MAIN EMBEDDING PIPELINE
# =============================================================================

def run_embedding_pipeline(run_id: str, embedding_model: str, embedding_dimension: int, pinecone_index_name: str, pinecone_environment: str) -> None:
    """
    Run the complete embedding pipeline for an experiment.
    
    Args:
        run_id: Run ID for the experiment
        embedding_model: Model name for creating embeddings
        embedding_dimension: Dimension of embeddings
    """
    try:
        logger.info(f"Starting embedding pipeline for run {run_id}")
        log_workflow_step(run_id, "embedding", "started")
        
        # Load chunks from previous step
        chunks_data = load_artifacts(run_id, "chunks")
        if not chunks_data or 'chunks' not in chunks_data:
            raise ValueError("No chunks found. Run chunking pipeline first.")
        
        chunks = chunks_data['chunks']
        logger.info(f"Loaded {len(chunks)} chunks for embedding")
        
        # Initialize Vertex AI
        initialize_vertex_ai()
        
        # Extract text from chunks
        texts = [chunk['chunk_text'] for chunk in chunks]
        
        # Create embeddings
        logger.info("Creating embeddings...")
        start_time = time.time()
        embeddings = create_embeddings_batch(texts, embedding_model, batch_size=100) 
        embedding_time = time.time() - start_time
        
        if len(embeddings) != len(chunks):
            raise ValueError(f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings")
        
        # Initialize Pinecone
        index = initialize_pinecone(embedding_dimension, pinecone_index_name, pinecone_environment)
        
        # Upsert vectors to Pinecone
        logger.info("Upserting vectors to Pinecone...")
        upsert_start_time = time.time()
        upsert_vectors_to_pinecone(chunks, embeddings, index)
        upsert_time = time.time() - upsert_start_time
        
        # Calculate statistics
        embedding_stats = {
            'total_chunks': len(chunks),
            'total_embeddings': len(embeddings),
            'embedding_time_seconds': embedding_time,
            'upsert_time_seconds': upsert_time,
            'total_time_seconds': embedding_time + upsert_time,
            'average_embedding_time_per_chunk': embedding_time / len(chunks) if chunks else 0,
            'embedding_dimension': len(embeddings[0]) if embeddings else 0
        }
        
        # Save metadata
        save_embedding_metadata(run_id, embedding_stats, embedding_model, embedding_dimension, pinecone_index_name)
        
        # Log metrics
        log_metrics(run_id, {
            'embedding_stats': embedding_stats,
            'pinecone_index_name': pinecone_index_name,
            'embedding_model': embedding_model
        })
        
        log_workflow_step(run_id, "embedding", "completed", f"Created {len(embeddings)} embeddings in {embedding_time:.2f}s")
        logger.info(f"Embedding pipeline completed successfully for run {run_id}")
        
        # Log summary
        logger.info("=== EMBEDDING PIPELINE SUMMARY ===")
        logger.info(f"Total chunks processed: {len(chunks)}")
        logger.info(f"Embedding time: {embedding_time:.2f} seconds")
        logger.info(f"Upsert time: {upsert_time:.2f} seconds")
        logger.info(f"Average time per chunk: {embedding_stats['average_embedding_time_per_chunk']:.3f} seconds")
        logger.info(f"Pinecone index: {pinecone_index_name}")
        
    except Exception as e:
        logger.error(f"Embedding pipeline failed for run {run_id}: {e}")
        log_workflow_step(run_id, "embedding", "failed", str(e))
        raise

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function to run the embedding pipeline.
    This is typically called from the experiment entry point.
    """
    try:
        # For standalone execution, use the latest run ID
        from config import get_latest_run_id
        
        run_id = get_latest_run_id()
        if not run_id:
            raise ValueError("No experiment runs found. Run chunking pipeline first.")
        
        logger.info(f"Using latest run: {run_id}")
        
        # Run embedding pipeline
        run_embedding_pipeline(run_id)
        
        logger.info("Embedding pipeline completed successfully.")
        
    except Exception as e:
        logger.error(f"Embedding pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
