"""
Utility functions for the Typeform RAG Prototype project.

This module contains logging, artifact management, and data profiling logic for the workflow.
"""

import json
import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from config import construct_path

# Configure logging
def setup_logging(run_id: str = None):
    """
    Setup comprehensive logging to both console and trace file.
    
    Args:
        run_id: Optional run ID for experiment-specific logging
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Trace file handler (if run_id provided)
    if run_id:
        trace_log_path = construct_path('full_trace_log', run_id)
        if trace_log_path:
            file_handler = logging.FileHandler(trace_log_path, mode='a', encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
    
    return logging.getLogger(__name__)

# Module-level logger
logger = logging.getLogger(__name__)

# =============================================================================
# ARTIFACT MANAGEMENT FUNCTIONS
# =============================================================================

def get_artifact_path(run_id: str, artifact_type: str, filename: str = None) -> tuple[str, str]:
    """
    Get directory and filepath for artifact type.
    
    Args:
        run_id: Run ID for the experiment
        artifact_type: Type of artifact (chunks, config, etc.)
        filename: Optional custom filename
        
    Returns:
        tuple: (artifacts_dir, filepath)
    """
    if artifact_type == "chunks":
        artifacts_dir = construct_path('chunks_artifacts_dir', run_id)
        if not filename:
            filename = "chunks.json"
    elif artifact_type == "chunk_stats":
        artifacts_dir = construct_path('chunks_summary_dir', run_id)
        if not filename:
            filename = "chunk_stats.csv"
    elif artifact_type == "embeddings_metadata":
        artifacts_dir = construct_path('vector_store_dir', run_id)
        if not filename:
            filename = "embeddings_model_metadata.json"
    else:
        raise ValueError(f"Unknown artifact_type: {artifact_type}")
    
    if not artifacts_dir:
        raise ValueError(f"Could not construct {artifact_type} directory path")
    
    filepath = os.path.join(artifacts_dir, filename)
    return artifacts_dir, filepath

def save_artifacts(data: Dict[str, Any], run_id: str, artifact_type: str, filename: str = None) -> str:
    """
    Save artifacts to experiment directory with proper structure.
    
    Args:
        data: Data to save
        run_id: Run ID for the experiment
        artifact_type: Type of artifact (chunks, config, etc.)
        filename: Optional custom filename
        
    Returns:
        str: Path to saved file
    """
    try:
        artifacts_dir, filepath = get_artifact_path(run_id, artifact_type, filename)
        
        # Extract filename from filepath for extension checking
        filename = os.path.basename(filepath)
        
        # Save based on file extension
        if filename.endswith('.json'):
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str, ensure_ascii=False)
        elif filename.endswith('.csv'):
            if isinstance(data, pd.DataFrame):
                data.to_csv(filepath, index=False)
            else:
                # Convert dict to DataFrame if needed
                df = pd.DataFrame(data)
                df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unsupported file extension for {filename}")
        
        logger.info(f"Saved {artifact_type} artifact: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Error saving {artifact_type} artifact: {e}")
        raise

def load_artifacts(run_id: str, artifact_type: str, filename: str = None) -> Any:
    """
    Load artifacts from experiment directory.
    
    Args:
        run_id: Run ID for the experiment
        artifact_type: Type of artifact to load
        filename: Optional custom filename
        
    Returns:
        Loaded data (dict, DataFrame, etc.)
    """
    try:
        artifacts_dir, filepath = get_artifact_path(run_id, artifact_type, filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"Artifact file not found: {filepath}")
            return None
        
        # Extract filename from filepath for extension checking
        filename = os.path.basename(filepath)
        
        # Load based on file extension
        if filename.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif filename.endswith('.csv'):
            return pd.read_csv(filepath)
        else:
            raise ValueError(f"Unsupported file extension for {filename}")
        
    except Exception as e:
        logger.error(f"Error loading {artifact_type} artifact: {e}")
        return None

# =============================================================================
# DATA PROFILING FUNCTIONS
# =============================================================================

def calculate_text_statistics(values: List[int]) -> Dict[str, float]:
    """
    Calculate standard statistics for a list of numeric values.
    
    Args:
        values: List of numeric values to analyze
        
    Returns:
        dict: Statistics including mean, std, min, max, median, q25, q75
    """
    if not values:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0,
            'max': 0,
            'median': 0.0,
            'q25': 0.0,
            'q75': 0.0
        }
    
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': int(np.min(values)),
        'max': int(np.max(values)),
        'median': float(np.median(values)),
        'q25': float(np.percentile(values, 25)),
        'q75': float(np.percentile(values, 75))
    }

def analyze_documents(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze document collection and generate statistics.
    
    Args:
        documents: List of document dictionaries with 'text' and metadata
        
    Returns:
        dict: Document analysis statistics
    """
    try:
        if not documents:
            return {}
        
        # Extract text lengths
        text_lengths = [len(doc.get('text', '')) for doc in documents]
        word_counts = [len(doc.get('text', '').split()) for doc in documents]
        
        # Calculate statistics using helper function
        stats = {
            'total_documents': len(documents),
            'text_length_stats': calculate_text_statistics(text_lengths),
            'word_count_stats': calculate_text_statistics(word_counts)
        }
        
        # Document metadata analysis
        if documents:
            sample_doc = documents[0]
            stats['metadata_fields'] = list(sample_doc.keys())
            stats['has_url'] = 'full_doc_url' in sample_doc
            stats['has_timestamp'] = 'read_timestamp' in sample_doc
        
        logger.info(f"Document analysis completed: {len(documents)} documents analyzed")
        return stats
        
    except Exception as e:
        logger.error(f"Error analyzing documents: {e}")
        return {}

def analyze_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze chunk collection and generate statistics.
    
    Args:
        chunks: List of chunk dictionaries with 'chunk_text' and metadata
        
    Returns:
        dict: Chunk analysis statistics
    """
    try:
        if not chunks:
            return {}
        
        # Extract chunk lengths
        chunk_lengths = [len(chunk.get('chunk_text', '')) for chunk in chunks]
        chunk_word_counts = [len(chunk.get('chunk_text', '').split()) for chunk in chunks]
        
        # Calculate statistics using helper function
        stats = {
            'total_chunks': len(chunks),
            'chunk_length_stats': calculate_text_statistics(chunk_lengths),
            'chunk_word_count_stats': calculate_text_statistics(chunk_word_counts)
        }
        
        # Chunk metadata analysis
        if chunks:
            sample_chunk = chunks[0]
            stats['metadata_fields'] = list(sample_chunk.keys())
            stats['has_parent_doc'] = 'parent_document_id' in sample_chunk
            stats['has_position'] = 'chunk_position_index' in sample_chunk
        
        # Document distribution analysis
        parent_docs = [chunk.get('parent_document_id') for chunk in chunks]
        unique_docs = set(parent_docs)
        stats['unique_parent_documents'] = len(unique_docs)
        stats['chunks_per_document'] = {
            'mean': float(len(chunks) / len(unique_docs)) if unique_docs else 0.0,
            'min': int(min([parent_docs.count(doc) for doc in unique_docs])) if unique_docs else 0,
            'max': int(max([parent_docs.count(doc) for doc in unique_docs])) if unique_docs else 0
        }
        
        logger.info(f"Chunk analysis completed: {len(chunks)} chunks analyzed")
        return stats
        
    except Exception as e:
        logger.error(f"Error analyzing chunks: {e}")
        return {}

def save_profile_to_csv(items: List[Dict[str, Any]], fields_extractor: callable, output_path: str) -> None:
    """
    Generic function to save profile data to CSV.
    
    Args:
        items: List of items to profile
        fields_extractor: Function that extracts fields from each item
        output_path: Path to save the CSV file
    """
    try:
        # Create DataFrame with extracted data
        profile_data = []
        for item in items:
            profile_data.append(fields_extractor(item))
        
        df = pd.DataFrame(profile_data)
        df.to_csv(output_path, index=False)
        logger.info(f"Profile saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving profile: {e}")
        raise

def save_document_profile(documents: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save document profile as CSV for analysis.
    
    Args:
        documents: List of document dictionaries
        output_path: Path to save the CSV file
    """
    def extract_doc_fields(doc):
        return {
            'doc_id': doc.get('doc_id', ''),
            'doc_title': doc.get('doc_title', ''),
            'text_length': len(doc.get('text', '')),
            'word_count': len(doc.get('text', '').split()),
            'read_timestamp': doc.get('read_timestamp', ''),
            'full_doc_url': doc.get('full_doc_url', '')
        }
    
    save_profile_to_csv(documents, extract_doc_fields, output_path)

def save_chunk_profile(chunks: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save chunk profile as CSV for analysis.
    
    Args:
        chunks: List of chunk dictionaries
        output_path: Path to save the CSV file
    """
    def extract_chunk_fields(chunk):
        return {
            'chunk_id': chunk.get('chunk_id', ''),
            'parent_document_id': chunk.get('parent_document_id', ''),
            'chunk_position_index': chunk.get('chunk_position_index', 0),
            'chunk_length': len(chunk.get('chunk_text', '')),
            'word_count': len(chunk.get('chunk_text', '').split())
        }
    
    save_profile_to_csv(chunks, extract_chunk_fields, output_path)

# =============================================================================
# LOGGING FUNCTIONS
# =============================================================================

def log_workflow_step(run_id: str, step: str, status: str, message: str = "", data: Dict[str, Any] = None) -> None:
    """
    Log a workflow step to the trace log.
    
    Args:
        run_id: Run ID for the experiment
        step: Step name
        status: Step status (started, completed, failed)
        message: Optional message
        data: Optional data dictionary to log
    """
    try:
        # Get trace log path
        trace_log_path = construct_path('full_trace_log', run_id)
        if not trace_log_path:
            return
        
        # Create log entry
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {step}: {status}"
        if message:
            log_entry += f" - {message}"
        if data:
            log_entry += f" - Data: {json.dumps(data, default=str)}"
        log_entry += "\n"
        
        # Append to log file
        with open(trace_log_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        logger.info(f"Logged workflow step: {step} - {status}")
        
    except Exception as e:
        logger.error(f"Error logging workflow step: {e}")

def append_to_json_log(log_path: str, entry_data: Dict[str, Any], add_timestamp: bool = True) -> None:
    """
    Append entry to JSON log file.
    
    Args:
        log_path: Path to the JSON log file
        entry_data: Data to append to the log
        add_timestamp: Whether to add timestamp to the entry
    """
    try:
        # Add timestamp if requested
        if add_timestamp:
            entry_data['timestamp'] = datetime.now().isoformat()
        
        # Load existing log or create new list
        if log_path and os.path.exists(log_path):
            with open(log_path, 'r', encoding='utf-8') as f:
                log_entries = json.load(f)
        else:
            log_entries = []
        
        # Append new entry
        log_entries.append(entry_data)
        
        # Save updated log
        if log_path:
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log_entries, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Appended entry to JSON log: {log_path}")
        
    except Exception as e:
        logger.error(f"Error appending to JSON log: {e}")
        raise

def log_retrieval_trace(run_id: str, query: str, retrieved_chunks: List[Dict], latency_ms: float, tokens_used: int = None) -> None:
    """
    Log retrieval trace for analysis.
    
    Args:
        run_id: Run ID for the experiment
        query: User query
        retrieved_chunks: List of retrieved chunks with metadata
        latency_ms: Retrieval latency in milliseconds
        tokens_used: Optional token count
    """
    try:
        trace_data = {
            'query': query,
            'num_chunks_retrieved': len(retrieved_chunks),
            'latency_ms': latency_ms,
            'tokens_used': tokens_used,
            'chunk_ids': [chunk.get('chunk_id', '') for chunk in retrieved_chunks],
            'chunk_scores': [chunk.get('score', 0.0) for chunk in retrieved_chunks]
        }
        
        # Get log path and append entry
        traces_path = construct_path('retrieval_traces_json', run_id)
        append_to_json_log(traces_path, trace_data)
        
        logger.info(f"Logged retrieval trace: {len(retrieved_chunks)} chunks retrieved in {latency_ms:.2f}ms")
        
    except Exception as e:
        logger.error(f"Error logging retrieval trace: {e}")

def log_metrics(run_id: str, metrics: Dict[str, Any]) -> None:
    """
    Log metrics for tracking and analysis.
    
    Args:
        run_id: Run ID for the experiment
        metrics: Dictionary of metrics to log
    """
    try:
        metrics_data = {
            'metrics': metrics
        }
        
        # Get log path and append entry
        metrics_path = construct_path('metrics_tracking_json', run_id)
        append_to_json_log(metrics_path, metrics_data)
        
        logger.info(f"Logged metrics: {list(metrics.keys())}")
        
    except Exception as e:
        logger.error(f"Error logging metrics: {e}")

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def validate_chunks(chunks: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """
    Validate chunk structure and content.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        if not chunks:
            return False, "No chunks provided"
        
        required_fields = ['chunk_id', 'chunk_text', 'parent_document_id']
        
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                return False, f"Chunk {i+1} is not a dictionary"
            
            for field in required_fields:
                if field not in chunk:
                    return False, f"Chunk {i+1} missing required field: {field}"
            
            if not chunk['chunk_text'].strip():
                return False, f"Chunk {i+1} has empty text content"
        
        logger.info(f"Chunk validation passed: {len(chunks)} chunks validated")
        return True, "All chunks validated successfully"
        
    except Exception as e:
        logger.error(f"Error validating chunks: {e}")
        return False, f"Validation error: {str(e)}"
