"""
Document chunking module for Typeform RAG prototype.

This module loads cleaned documents and splits them into semantically coherent chunks
using LangChain's RecursiveCharacterTextSplitter with configurable parameters.
"""

import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import config and utils
import sys
from config import (
    CLEAN_DOCS_ARTIFACTS_DIR, construct_path
)
from src.utils import (
    analyze_chunks, save_chunk_profile, save_artifacts, 
    validate_chunks, log_workflow_step
)

# =============================================================================
# CHUNKING FUNCTIONS
# =============================================================================

def load_cleaned_documents() -> List[Dict[str, Any]]:
    """
    Load all cleaned documents from the clean_docs directory.
    
    Returns:
        list: List of document dictionaries
    """
    documents = []
    
    if not os.path.exists(CLEAN_DOCS_ARTIFACTS_DIR):
        logger.error(f"Clean documents directory not found: {CLEAN_DOCS_ARTIFACTS_DIR}")
        return documents
    
    # Get all JSON files in the directory
    json_files = [f for f in os.listdir(CLEAN_DOCS_ARTIFACTS_DIR) if f.endswith('.json')]
    
    if not json_files:
        logger.warning(f"No JSON files found in {CLEAN_DOCS_ARTIFACTS_DIR}")
        return documents
    
    logger.info(f"Found {len(json_files)} cleaned documents to load")
    
    for json_file in json_files:
        file_path = os.path.join(CLEAN_DOCS_ARTIFACTS_DIR, json_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                document = json.load(f)
            documents.append(document)
            logger.info(f"Loaded document: {document.get('doc_id', 'unknown')}")
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")
            continue
    
    logger.info(f"Successfully loaded {len(documents)} documents")
    return documents

def create_text_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """
    Create a RecursiveCharacterTextSplitter with specified parameters.
    
    Args:
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        RecursiveCharacterTextSplitter: Configured text splitter
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " "],
        is_separator_regex=False
    )

def chunk_document(document: Dict[str, Any], text_splitter: RecursiveCharacterTextSplitter) -> List[Dict[str, Any]]:
    """
    Chunk a single document into smaller pieces.
    
    Args:
        document: Document dictionary with 'text' and metadata
        text_splitter: Configured text splitter
        
    Returns:
        list: List of chunk dictionaries
    """
    doc_id = document.get('doc_id', 'unknown')
    doc_text = document.get('text', '')
    
    if not doc_text.strip():
        logger.warning(f"Document {doc_id} has no text content")
        return []
    
    # Split the text into chunks
    text_chunks = text_splitter.split_text(doc_text)
    
    # Create chunk objects with metadata
    chunks = []
    for i, chunk_text in enumerate(text_chunks):
        chunk_id = f"{doc_id}_chunk_{i:03d}"
        
        chunk = {
            'chunk_id': chunk_id,
            'chunk_text': chunk_text.strip(),
            'parent_document_id': doc_id,
            'chunk_position_index': i,
            'chunk_length': len(chunk_text.strip()),
            'word_count': len(chunk_text.strip().split()),
            # Copy relevant document metadata
            'doc_title': document.get('doc_title', ''),
            'full_doc_url': document.get('full_doc_url', ''),
            'language': document.get('language', 'en'),
            'read_timestamp': document.get('read_timestamp', '')
        }
        
        chunks.append(chunk)
    
    logger.info(f"Created {len(chunks)} chunks for document {doc_id}")
    return chunks

def chunk_all_documents(documents: List[Dict[str, Any]], chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    """
    Chunk all documents using the specified parameters with parallel processing.
    
    Args:
        documents: List of document dictionaries
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        list: List of all chunks from all documents
    """
    logger.info(f"Starting parallel chunking process with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    # Create text splitter
    text_splitter = create_text_splitter(chunk_size, chunk_overlap)
    
    all_chunks = []
    
    # Use ThreadPoolExecutor for parallel document processing
    max_workers = min(4, len(documents))  # Use up to 4 workers, but not more than documents
    logger.info(f"Using {max_workers} workers for parallel chunking")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all documents for processing
        future_to_doc = {
            executor.submit(chunk_document, document, text_splitter): document 
            for document in documents
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_doc):
            document = future_to_doc[future]
            try:
                chunks = future.result()
                all_chunks.extend(chunks)
                doc_id = document.get('doc_id', 'unknown')
                logger.info(f"Completed chunking document {doc_id}: {len(chunks)} chunks")
            except Exception as e:
                doc_id = document.get('doc_id', 'unknown')
                logger.error(f"Error chunking document {doc_id}: {e}")
                continue
    
    logger.info(f"Parallel chunking completed: {len(all_chunks)} total chunks created from {len(documents)} documents")
    return all_chunks

def save_chunks(chunks: List[Dict[str, Any]], run_id: str) -> str:
    """
    Save chunks to experiment directory.
    
    Args:
        chunks: List of chunk dictionaries
        run_id: Run ID for the experiment
        
    Returns:
        str: Path to saved chunks file
    """
    try:
        # Validate chunks before saving
        is_valid, error_msg = validate_chunks(chunks)
        if not is_valid:
            raise ValueError(f"Chunk validation failed: {error_msg}")
        
        # Save chunks as JSON
        chunks_data = {
            'chunks': chunks,
            'total_chunks': len(chunks),
            'created_at': datetime.now().isoformat()
        }
        
        file_path = save_artifacts(chunks_data, run_id, "chunks")
        logger.info(f"Saved {len(chunks)} chunks to: {file_path}")
        return file_path
        
    except Exception as e:
        logger.error(f"Error saving chunks: {e}")
        raise

def generate_chunk_analysis(chunks: List[Dict[str, Any]], run_id: str) -> None:
    """
    Generate and save chunk analysis statistics.
    
    Args:
        chunks: List of chunk dictionaries
        run_id: Run ID for the experiment
    """
    try:
        # Analyze chunks
        analysis = analyze_chunks(chunks)
        
        # Save chunk profile CSV
        chunk_stats_path = os.path.join(
            construct_path('chunks_summary_dir', run_id), 
            "chunk_stats.csv"
        )
        save_chunk_profile(chunks, chunk_stats_path)
        
        # Save analysis summary
        analysis_path = os.path.join(
            construct_path('chunks_summary_dir', run_id), 
            "chunk_analysis.json"
        )
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Generated chunk profile: {chunk_stats_path}")
        logger.info(f"Generated analysis summary: {analysis_path}")
        
        # Log summary statistics
        logger.info("=== CHUNK ANALYSIS SUMMARY ===")
        logger.info(f"Total chunks: {analysis.get('total_chunks', 0)}")
        if 'chunk_length_stats' in analysis:
            stats = analysis['chunk_length_stats']
            logger.info(f"Chunk length - Mean: {stats['mean']:.0f}, Min: {stats['min']}, Max: {stats['max']}")
        if 'chunk_word_count_stats' in analysis:
            stats = analysis['chunk_word_count_stats']
            logger.info(f"Word count - Mean: {stats['mean']:.0f}, Min: {stats['min']}, Max: {stats['max']}")
        if 'unique_parent_documents' in analysis:
            logger.info(f"Unique parent documents: {analysis['unique_parent_documents']}")
        
    except Exception as e:
        logger.error(f"Error generating chunk analysis: {e}")
        raise

# =============================================================================
# MAIN CHUNKING PIPELINE
# =============================================================================

def run_chunking_pipeline(run_id: str, chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    """
    Run the complete chunking pipeline for an experiment.
    
    Args:
        run_id: Run ID for the experiment
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        list: List of all chunks created
    """
    try:
        logger.info(f"Starting chunking pipeline for run {run_id}")
        log_workflow_step(run_id, "chunking", "started", f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        
        # Load cleaned documents
        documents = load_cleaned_documents()
        if not documents:
            raise ValueError("No cleaned documents found. Run clean_knowledge.py first.")
        
        # Chunk all documents
        chunks = chunk_all_documents(documents, chunk_size, chunk_overlap)
        if not chunks:
            raise ValueError("No chunks were created")
        
        # Save chunks
        save_chunks(chunks, run_id)
        
        # Generate analysis
        generate_chunk_analysis(chunks, run_id)
        
        log_workflow_step(run_id, "chunking", "completed", f"Created {len(chunks)} chunks")
        logger.info(f"Chunking pipeline completed successfully for run {run_id}")
        
        return chunks
        
    except Exception as e:
        logger.error(f"Chunking pipeline failed for run {run_id}: {e}")
        log_workflow_step(run_id, "chunking", "failed", str(e))
        raise

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function to run the chunking pipeline.
    This is typically called from the experiment entry point.
    """
    raise RuntimeError(
        "chunk_documents.py cannot be run standalone. "
        "All hyperparameters (chunk_size, chunk_overlap) must be explicitly set. "
        "Use dev_interactive_run.py or call run_chunking_pipeline() with explicit parameters."
    )

if __name__ == "__main__":
    main()
