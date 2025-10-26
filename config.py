"""
Configuration file for the Typeform RAG Prototype project.

This file centralizes all project paths and settings. Users should modify
the PROJECT_ROOT and credentials to match their environment.
"""

import os
import sys
import json
import uuid
import yaml
from datetime import datetime, timezone
from typing import Optional, Dict, Any

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()  # Loads .env file if it exists

# =============================================================================
# USER CONFIGURATION - MODIFY THESE VALUES
# =============================================================================

# Project root path - from environment variable with auto-detection fallback
import os
PROJECT_ROOT = os.getenv("PROJECT_ROOT")
if not PROJECT_ROOT:
    # Auto-detect if not set
    if os.path.exists('/app'):
        PROJECT_ROOT = "/app"
    else:
        PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Configuration from environment variables (with fallbacks)
import os

# Google Cloud credentials file (expected in PROJECT_ROOT/credentials)
GOOGLE_CREDENTIALS_FILE = os.getenv("GOOGLE_CREDENTIALS_FILE", "typeform-rag-credentials.json")

# Google Cloud Project ID
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "your-gcp-project-id")

# Pinecone Configuration
PINECONE_API_KEY_FILE = os.getenv("PINECONE_API_KEY_FILE", "PINECONE_API_KEY")


# =============================================================================
# COMPUTED PATHS - AUTOMATICALLY GENERATED FROM PROJECT_ROOT
# =============================================================================

# Add project root to sys path for easy imports
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Source directory
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

# Knowledge base directories
KNOWLEDGEBASE_DIR = os.path.join(PROJECT_ROOT, "knowledgebase")
RAW_DOCS_DIR = os.path.join(KNOWLEDGEBASE_DIR, "raw_docs")
CLEAN_DOCS_DIR = os.path.join(KNOWLEDGEBASE_DIR, "clean_docs")
CLEAN_DOCS_ARTIFACTS_DIR = os.path.join(CLEAN_DOCS_DIR, "docs")
CLEAN_DOCS_SUMMARY_DIR = os.path.join(CLEAN_DOCS_DIR, "summary")

# Experiments directory (holds all experiment runs)
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")

# Credentials directory
CREDENTIALS_DIR = os.path.join(PROJECT_ROOT, "credentials")

# Prompts directory
PROMPTS_DIR = os.path.join(PROJECT_ROOT, "prompts")

# Evaluation base directory
EVALUATION_BASE_DIR = os.path.join(PROJECT_ROOT, "evaluation-base")

# Set Google Cloud credentials environment variable
if os.path.exists(os.path.join(CREDENTIALS_DIR, GOOGLE_CREDENTIALS_FILE)):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(CREDENTIALS_DIR, GOOGLE_CREDENTIALS_FILE)

# =============================================================================
# RUN ID MANAGEMENT
# =============================================================================

def generate_run_id() -> str:
    """
    Generate a new run ID with timestamp (including timezone) and UUID format.
    Format: YYYYMMDD_HHMMSS_TZ_<UUID>
    
    Returns:
        str: Unique run ID with timestamp and UUID
    """
    # Get current time with timezone
    now = datetime.now(timezone.utc)
    timestamp = now.strftime('%Y%m%d_%H%M%S_UTC')
    unique_id = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
    return f"{timestamp}_{unique_id}"

def get_latest_run_id() -> Optional[str]:
    """
    Get the most recent run ID from experiments directory.
    
    Returns:
        str: Latest run ID, or None if no runs exist
    """
    if not os.path.exists(EXPERIMENTS_DIR):
        return None
    
    run_dirs = [d for d in os.listdir(EXPERIMENTS_DIR) 
                if os.path.isdir(os.path.join(EXPERIMENTS_DIR, d))]
    
    if not run_dirs:
        return None
    
    # Sort by timestamp (embedded in directory name) and return the latest
    run_dirs.sort(reverse=True)
    return run_dirs[0]

def validate_run_id(run_id: str) -> bool:
    """
    Validate that a run ID exists and has proper structure.
    
    Args:
        run_id: Run ID to validate
        
    Returns:
        bool: True if run ID is valid and directory exists
    """
    if not run_id:
        return False
    
    run_path = os.path.join(EXPERIMENTS_DIR, run_id)
    return os.path.exists(run_path)

def create_run_directory(run_id: str) -> str:
    """
    Create the directory structure for a new experiment run.
    
    Structure:
        experiments/{run_id}/
            ├── chunks/
            │   ├── artifacts/
            │   └── summary/
            ├── vector-store-data/
            ├── logs/
            └── config.yaml
    
    Args:
        run_id: Unique run identifier
        
    Returns:
        str: Path to the created run directory
    """
    run_dir = os.path.join(EXPERIMENTS_DIR, run_id)
    
    # Create main run directory
    os.makedirs(run_dir, exist_ok=True)
    
    # Create subdirectories following the plan structure
    subdirs = [
        'chunks/artifacts',
        'chunks/summary',
        'vector-store-data',
        'logs',
        'answers_to_testQ'
    ]
    
    for subdir in subdirs:
        os.makedirs(os.path.join(run_dir, subdir), exist_ok=True)
    
    return run_dir

# =============================================================================
# PATH CONSTRUCTION FUNCTIONS
# =============================================================================

def construct_path(object_name: str, run_id: Optional[str] = None) -> Optional[str]:
    """
    Construct paths for various objects in the project.
    
    Args:
        object_name: Name of the object/path to construct
        run_id: Optional run ID for experiment-specific paths
    
    Returns:
        str: Full path to the object, or None if not found
    """
    # Handle paths that don't require run_id
    if run_id is None:
        dictionary_of_objects = {
            # Knowledge base paths
            'raw_docs_dir': RAW_DOCS_DIR,
            'clean_docs_dir': CLEAN_DOCS_DIR,
            'clean_docs_artifacts_dir': CLEAN_DOCS_ARTIFACTS_DIR,
            'clean_docs_summary_dir': CLEAN_DOCS_SUMMARY_DIR,
            'doc_stats_csv': os.path.join(CLEAN_DOCS_SUMMARY_DIR, "doc_stats.csv"),
            
            # Project directories
            'credentials_dir': CREDENTIALS_DIR,
            'experiments_dir': EXPERIMENTS_DIR,
            'src_dir': SRC_DIR,
            'prompts_dir': PROMPTS_DIR,
            'evaluation_base_dir': EVALUATION_BASE_DIR,
            
            # Credentials files
            'credentials_file': os.path.join(CREDENTIALS_DIR, GOOGLE_CREDENTIALS_FILE),
            'pinecone_api_key_file': os.path.join(CREDENTIALS_DIR, PINECONE_API_KEY_FILE),
        }
    else:
        # Validate run_id exists
        if not validate_run_id(run_id):
            raise ValueError(f"Invalid run_id: {run_id}. Run directory does not exist.")
        
        run_dir = os.path.join(EXPERIMENTS_DIR, run_id)
        
        # Handle paths that require run_id (experiment-specific)
        dictionary_of_objects = {
            'run_dir': run_dir,
            'config_yaml': os.path.join(run_dir, "config.yaml"),
            
            # Chunks paths
            'chunks_dir': os.path.join(run_dir, "chunks"),
            'chunks_artifacts_dir': os.path.join(run_dir, "chunks", "artifacts"),
            'chunks_summary_dir': os.path.join(run_dir, "chunks", "summary"),
            'chunks_file': os.path.join(run_dir, "chunks", "artifacts", "chunks.json"),
            'chunk_stats_csv': os.path.join(run_dir, "chunks", "summary", "chunk_stats.csv"),
            
            # Vector store paths
            'vector_store_dir': os.path.join(run_dir, "vector-store-data"),
            'embeddings_metadata': os.path.join(run_dir, "vector-store-data", "embeddings_model_metadata.json"),
            
            # Logs paths
            'logs_dir': os.path.join(run_dir, "logs"),
            'full_trace_log': os.path.join(run_dir, "logs", "full_trace.log"),
            'retrieval_traces_json': os.path.join(run_dir, "logs", "retrieval_traces.json"),
            'metrics_tracking_json': os.path.join(run_dir, "logs", "metrics_tracking.json"),
            
            # Test results paths
            'answers_to_testQ_dir': os.path.join(run_dir, "answers_to_testQ"),
        }
    
    return dictionary_of_objects.get(object_name, None)

def get_pinecone_api_key() -> str:
    """
    Read Pinecone API key from credentials file.
    
    Returns:
        str: Pinecone API key
        
    Raises:
        FileNotFoundError: If API key file not found
    """
    api_key_path = os.path.join(CREDENTIALS_DIR, PINECONE_API_KEY_FILE)
    try:
        with open(api_key_path, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Pinecone API key file not found at {api_key_path}")
    except Exception as e:
        raise Exception(f"Error reading Pinecone API key: {e}")

# =============================================================================
# EXPERIMENT CONFIGURATION MANAGEMENT
# =============================================================================

def save_experiment_config(run_id: str, config: Dict[str, Any]) -> None:
    """
    Save experiment configuration to YAML file.
    
    Args:
        run_id: Run ID for the experiment
        config: Configuration dictionary to save
    """
    config_path = construct_path('config_yaml', run_id)
    if config_path:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def load_experiment_config(run_id: str) -> Dict[str, Any]:
    """
    Load experiment configuration from YAML file.
    
    Args:
        run_id: Run ID for the experiment
        
    Returns:
        dict: Configuration dictionary, or empty dict if file doesn't exist
    """
    config_path = construct_path('config_yaml', run_id)
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def create_experiment_config(chunk_size: int, chunk_overlap: int,
                            embedding_model: str, embedding_dimension: int,
                            top_k_retrieval: int, llm_model: str,
                            llm_temperature: float, llm_max_tokens: int,
                            pinecone_index_name: str, pinecone_environment: str) -> Dict[str, Any]:
    """
    Create experiment configuration with all required parameters.
    
    Args:
        chunk_size: Size of document chunks
        chunk_overlap: Overlap between chunks
        embedding_model: Model for creating embeddings
        embedding_dimension: Dimension of embeddings
        top_k_retrieval: Number of chunks to retrieve
        llm_model: LLM model for response generation
        llm_temperature: Temperature for LLM generation
        llm_max_tokens: Maximum tokens for LLM response
        pinecone_index_name: Pinecone index name
        pinecone_environment: Pinecone environment
    
    Returns:
        dict: Configuration with all hyperparameters
    """
    return {
        'experiment_info': {
            'run_id': None,  # Will be set when run is created
            'created_at': datetime.now(timezone.utc).isoformat(),
            'description': 'RAG prototype experiment'
        },
        'chunking': {
            'strategy': 'recursive',
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'separators': ["\n\n", "\n", ". ", " "]
        },
        'embedding': {
            'model': embedding_model,
            'dimension': embedding_dimension,
            'batch_size': 100
        },
        'retrieval': {
            'top_k': top_k_retrieval,
            'index_name': pinecone_index_name,
            'environment': pinecone_environment
        },
        'llm': {
            'model': llm_model,
            'temperature': llm_temperature,
            'max_tokens': llm_max_tokens
        }
    }

# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_project_structure():
    """
    Initialize the project directory structure if it doesn't exist.
    Creates all necessary directories for the project.
    """
    directories = [
        CREDENTIALS_DIR,
        RAW_DOCS_DIR,
        CLEAN_DOCS_ARTIFACTS_DIR,
        CLEAN_DOCS_SUMMARY_DIR,
        EXPERIMENTS_DIR,
        PROMPTS_DIR,
        EVALUATION_BASE_DIR,
        SRC_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Initialize project structure on import
initialize_project_structure()

