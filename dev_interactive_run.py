# %%
"""
Interactive entry point for Typeform RAG prototype.

This script provides a notebook-style interface for running the complete RAG pipeline
with configurable parameters and step-by-step execution.
"""

import os
import sys
import logging
from datetime import datetime
import importlib

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()  # Loads .env file if it exists

# Setup basic logging early for module reloading
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Reload all internal components to pick up latest code changes
logger.info("Reloading internal components...")
modules_to_reload = [
    'config',
    'src.utils',
    'src.knowledge_acquisition.clean_knowledge',
    'src.chunking.chunk_documents',
    'src.embedding.create_embeddings',
    'src.rag.retriever',
    'src.rag.generator',
    'src.evaluation.evaluation'
]

for module_name in modules_to_reload:
    try:
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
            logger.info(f"  Reloaded {module_name}")
        else:
            logger.info(f"  - {module_name} not yet loaded")
    except Exception as e:
        logger.info(f"  Failed to reload {module_name}: {e}")

logger.info("Component reload completed")

# Optional import for FastAPI server
try:
    import uvicorn
    from src.api.app import app
    UVICORN_AVAILABLE = True
except ImportError:
    UVICORN_AVAILABLE = False
    logger.info("Note: uvicorn not available - FastAPI server cell will be skipped")

# Import configuration and utilities
# Note: config.py automatically adds PROJECT_ROOT to sys.path
from config import (
    generate_run_id, create_run_directory, save_experiment_config,
    create_experiment_config, construct_path
)
from src.utils import setup_logging, log_workflow_step

# Import pipeline components
from src.knowledge_acquisition.clean_knowledge import main as run_clean_knowledge
from src.chunking.chunk_documents import run_chunking_pipeline
from src.embedding.create_embeddings import run_embedding_pipeline
from src.rag.retriever import RAGRetriever
from src.rag.generator import RAGGenerator
from src.evaluation.evaluation import analyze_response_quality

# %% [markdown]
# # Typeform RAG Prototype - Interactive Notebook
# 
# This notebook provides a step-by-step interface for running the complete RAG pipeline.
# Each cell can be run independently in VS Code's interactive Python mode, or run all
# cells sequentially by executing `python3 entrypoint.py`.
# 
# ## Workflow Overview:
# 1. **Configuration** - Set all parameters and execution flags
# 2. **Setup** - Initialize experiment run and logging
# 3. **Knowledge Cleaning** - Clean raw documents (run once)
# 4. **Chunking** - Split documents into chunks
# 5. **Embedding** - Create vector embeddings and load to Pinecone
# 6. **Test Retrieval** - Validate retrieval functionality
# 7. **Test RAG** - Test end-to-end RAG pipeline


# %% Cell 2: Configuration Parameters
# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================
# Modify these parameters to configure your RAG experiment
# These are the defaults - change them here to run different experiments

# Chunking parameters
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100

# Retrieval parameters
TOP_K = 6

# LLM parameters
TEMPERATURE = 0.1
MAX_TOKENS = 1024
PROMPT_VERSION = "detailed"

# Model selection
EMBEDDING_MODEL = "text-embedding-005"
EMBEDDING_DIMENSION = 768
LLM_MODEL = "gemini/gemini-2.5-flash"

# Pinecone Configuration
PINECONE_INDEX_NAME = "typeform-help-rag"
PINECONE_ENVIRONMENT = "us-east-1"

# Validate prompt version
from prompts.system_prompts import PROMPT_VERSIONS
if PROMPT_VERSION not in PROMPT_VERSIONS:
    available_versions = list(PROMPT_VERSIONS.keys())
    logger.error(f"ERROR: PROMPT_VERSION '{PROMPT_VERSION}' is not valid!")
    logger.error(f"Available prompt versions: {available_versions}")
    logger.error("Please set PROMPT_VERSION in dev_interactive_run.py to one of the available versions.")
    sys.exit(1)

# Test queries for validation (used in testing cells)
TEST_QUERIES = [
    {"query": "How do I create multi-language forms?", "file_identifier": "multi-lang"},
    {"query": "What is a multi-question page and how do I add one?", "file_identifier": "multiq"},
    {"query": "How do I sign into my typerform account?", "file_identifier": "signin"}
]

# Execution control flags
SKIP_CLEAN_KNOWLEDGE = False  # Set to False to clean raw docs (run once); set to True to skip cleaning if already done
SKIP_CHUNKING = False
SKIP_EMBEDDING = False
SKIP_TEST_RETRIEVAL = False
SKIP_TEST_RAG = False

logger.info("Configuration loaded:")
logger.info(f"  Chunk size: {CHUNK_SIZE}")
logger.info(f"  Chunk overlap: {CHUNK_OVERLAP}")
logger.info(f"  Top K: {TOP_K}")
logger.info(f"  Temperature: {TEMPERATURE}")
logger.info(f"  Prompt version: {PROMPT_VERSION}")

# %% Cell 3: Setup and Initialization
logger.info("=" * 60)
logger.info("CELL 3: SETUP AND INITIALIZATION")
logger.info("=" * 60)

# Generate run ID and create directory
run_id = generate_run_id()
create_run_directory(run_id)

# Setup logging
logger = setup_logging(run_id)

# Create experiment configuration with all required parameters
config = create_experiment_config(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    embedding_model=EMBEDDING_MODEL,
    embedding_dimension=EMBEDDING_DIMENSION,
    top_k_retrieval=TOP_K,
    llm_model=LLM_MODEL,
    llm_temperature=TEMPERATURE,
    llm_max_tokens=MAX_TOKENS,
    pinecone_index_name=PINECONE_INDEX_NAME,
    pinecone_environment=PINECONE_ENVIRONMENT
)
config['experiment_info']['run_id'] = run_id
config['llm']['prompt_version'] = PROMPT_VERSION

# Save configuration
save_experiment_config(run_id, config)

logger.info(f"Created experiment run: {run_id}")
logger.info(f"Configuration saved")
logger.info(f"Logging initialized")

# %% [markdown]
# ## Knowledge Cleaning
# 
# This step cleans and processes raw documents from the knowledge base.
# can run this once 
# - set `SKIP_CLEAN_KNOWLEDGE = True` in Cell 2 if you've already cleaned the docs
# - set `SKIP_CLEAN_KNOWLEDGE = False` in Cell 2 if you have not cleaned them and need to do it
# reruning the clean step is ok in the prototype since the number of articles is small

# %% Cell 4: Knowledge Cleaning
if not SKIP_CLEAN_KNOWLEDGE:
    logger.info("=" * 60)
    logger.info("CELL 4: KNOWLEDGE ACQUISITION AND CLEANING")
    logger.info("=" * 60)
    
    try:
        logger.info("Running knowledge cleaning pipeline...")
        run_clean_knowledge()
        logger.info("Knowledge cleaning completed successfully")
        
        # Check results
        from config import CLEAN_DOCS_ARTIFACTS_DIR
        if os.path.exists(CLEAN_DOCS_ARTIFACTS_DIR):
            json_files = [f for f in os.listdir(CLEAN_DOCS_ARTIFACTS_DIR) if f.endswith('.json')]
            logger.info(f"Found {len(json_files)} cleaned documents")
        
    except Exception as e:
        logger.info(f"Knowledge cleaning failed: {e}")
        raise
else:
    logger.info("Skipping knowledge cleaning (SKIP_CLEAN_KNOWLEDGE=True)")
    logger.info("Using existing cleaned documents")

# %% [markdown]
# ## Document Chunking
# 
# This step splits the cleaned documents into smaller chunks for embedding.
# Chunk size and overlap can be configured in Cell 2.

# %% Cell 5: Document Chunking
if not SKIP_CHUNKING:
    logger.info("=" * 60)
    logger.info("CELL 5: DOCUMENT CHUNKING")
    logger.info("=" * 60)
    
    try:
        logger.info(f"Running chunking pipeline...")
        logger.info(f"   Chunk size: {CHUNK_SIZE}")
        logger.info(f"   Chunk overlap: {CHUNK_OVERLAP}")
        
        chunks = run_chunking_pipeline(run_id, CHUNK_SIZE, CHUNK_OVERLAP)
        
        logger.info(f"Chunking completed: {len(chunks)} chunks created")
        
        # Show chunk statistics
        if chunks:
            chunk_lengths = [len(chunk['chunk_text']) for chunk in chunks]
            avg_length = sum(chunk_lengths) / len(chunk_lengths)
            logger.info(f"   Average chunk length: {avg_length:.0f} characters")
            logger.info(f"   Chunk length range: {min(chunk_lengths)} - {max(chunk_lengths)}")
        
    except Exception as e:
        logger.info(f"Chunking failed: {e}")
        raise
else:
    logger.info("Skipping chunking (SKIP_CHUNKING=True)")

# %% [markdown]
# ## Embedding and Vector Store
# 
# This step creates embeddings for the chunks and loads them into Pinecone.
# The embeddings will be used for semantic search during retrieval.

# %% Cell 6: Embedding and Vector Store
if not SKIP_EMBEDDING:
    logger.info("=" * 60)
    logger.info("CELL 6: EMBEDDING AND VECTOR STORE")
    logger.info("=" * 60)
    
    try:
        logger.info("Running embedding pipeline...")
        
        run_embedding_pipeline(run_id, EMBEDDING_MODEL, EMBEDDING_DIMENSION, PINECONE_INDEX_NAME, PINECONE_ENVIRONMENT)
        
        logger.info("Embedding pipeline completed successfully")
        logger.info("Vectors loaded to Pinecone")
        
    except Exception as e:
        logger.info(f"Embedding pipeline failed: {e}")
        raise
else:
    logger.info("Skipping embedding (SKIP_EMBEDDING=True)")

# %% [markdown]
# ## Test Retrieval
# 
# This step tests the retrieval functionality by running sample queries
# against the vector store to ensure chunks are being retrieved correctly.

# %% Cell 7: Test Retrieval
if not SKIP_TEST_RETRIEVAL:
    logger.info("=" * 60)
    logger.info("CELL 7: TEST RETRIEVAL")
    logger.info("=" * 60)
    
    try:
        # Initialize retriever
        retriever = RAGRetriever(run_id=run_id, embedding_model=EMBEDDING_MODEL, pinecone_index_name=PINECONE_INDEX_NAME)
        
        for i, test_query in enumerate(TEST_QUERIES, 1):
            query = test_query["query"]
            logger.info(f"\nTest Query {i}: '{query}'")
            
            # Retrieve context
            chunks = retriever.retrieve_context(query, top_k=3)
            
            if chunks:
                logger.info(f"   Retrieved {len(chunks)} chunks")
                logger.info(f"   Score range: {chunks[-1]['score']:.3f} - {chunks[0]['score']:.3f}")
                logger.info(f"   First chunk preview: {chunks[0]['chunk_text'][:100]}...")
            else:
                logger.info("   No chunks retrieved")
        
        logger.info(f"\n Retrieval testing completed")
        
    except Exception as e:
        logger.info(f" Retrieval testing failed: {e}")
        raise
else:
    logger.info("Skipping retrieval testing (SKIP_TEST_RETRIEVAL=True)")
    # Still initialize retriever for later use
    retriever = RAGRetriever(run_id=run_id, embedding_model=EMBEDDING_MODEL, pinecone_index_name=PINECONE_INDEX_NAME)

# %% [markdown]
# ## Test End-to-End RAG
# 
# This step tests the complete RAG pipeline by generating responses
# to sample queries using the retrieved context and LLM.

# %% Cell 8: Test End-to-End RAG
if not SKIP_TEST_RAG:
    logger.info("=" * 60)
    logger.info("CELL 8: TEST END-TO-END RAG")
    logger.info("=" * 60)
    
    try:
        # Initialize generator
        generator = RAGGenerator(run_id=run_id, model=LLM_MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
        
        # Get answers directory path from config
        answers_dir = construct_path("answers_to_testQ_dir", run_id)
        logger.info(f"Using answers directory: {answers_dir}")
        
        for i, test_query in enumerate(TEST_QUERIES, 1):
            query = test_query["query"]
            file_identifier = test_query["file_identifier"]
            logger.info(f"\nRAG Test {i}: '{query}'")
            
            # Retrieve context
            chunks = retriever.retrieve_context(query, top_k=TOP_K)
            
            # Generate response
            response_data = generator.generate_response(query, chunks, PROMPT_VERSION)
            
            # Analyze response quality
            quality_analysis = analyze_response_quality(response_data, chunks)
            
            logger.info(f"    Generated response in {response_data['generation_time_seconds']:.2f}s")
            logger.info(f"    Answer: {response_data['answer'][:200]}...")
            logger.info(f"    Citations: {len(response_data['citations'])}")
            
            if response_data['citations']:
                logger.info(f"   Citation IDs: {response_data['citations']}")
            
            # Format chunks for output
            chunks_list = [f"{chunk['chunk_id']} (score: {chunk['score']:.3f})" for chunk in chunks]
            
            # Create output content
            output_content = f"Question: {query}\n"
            output_content += f"Answer: {response_data['answer']}\n"
            output_content += f"chunks_retrieved: {chunks_list}\n"
            output_content += f"citations: {response_data['citations']}\n"
            output_content += f"\nResponse Quality Analysis:\n"
            for key, value in quality_analysis.items():
                output_content += f"  {key}: {value}\n"
            
            # Save to file
            output_file = os.path.join(answers_dir, f"{file_identifier}.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(output_content)
            
            logger.info(f"    Saved results to: {output_file}")
        
        logger.info(f"\n End-to-end RAG testing completed")
        
    except Exception as e:
        logger.info(f" RAG testing failed: {e}")
        raise
else:
    logger.info("Skipping RAG testing (SKIP_TEST_RAG=True)")
    # Still initialize generator for server use
    generator = RAGGenerator(run_id=run_id, model=LLM_MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)

# %%
