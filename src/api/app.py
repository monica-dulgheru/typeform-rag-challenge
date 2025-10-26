"""
FastAPI application for Typeform RAG prototype.

This module provides the REST API endpoint for the RAG chatbot.

Users can ONLY ask questions - the query is the only parameter that can be changed via the API - all other model parameters are set at 
model-build time, cannot be changed, and are tracked/retrieved using experiment IDs. 
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
import logging
import time
import os
import glob
import uuid
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import RAG components
from src.rag.retriever import RAGRetriever
from src.rag.generator import RAGGenerator
from src.evaluation.evaluation import analyze_response_quality
from config import load_experiment_config

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class QuestionRequest(BaseModel):
    """Request model for asking questions."""
    question: str

class SourceInfo(BaseModel):
    """Model for source information."""
    chunk_id: str
    doc_title: str
    score: float

class QuestionResponse(BaseModel):
    """Response model for question answers."""
    question: str
    answer: str
    sources: List[SourceInfo]
    citations: List[str]
    metadata: Dict[str, Any]

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    version: str

# =============================================================================
# LIFESPAN CONTEXT MANAGER
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    global retriever, generator, current_run_id, configured_prompt_version
    
    # Startup code
    try:
        logger.info("Initializing RAG components...")
        
        # Get run_id from environment variable or auto-detect latest experiment
        run_id = os.environ.get('RUN_ID')
        if run_id:
            logger.info(f"Using run_id from environment: {run_id}")
            current_run_id = run_id
        else:
            # Auto-detect latest experiment
            experiment_dirs = glob.glob('experiments/*')
            if experiment_dirs:
                latest_experiment = max(experiment_dirs, key=os.path.getmtime)
                run_id = os.path.basename(latest_experiment)
                logger.info(f"Auto-detected latest experiment: {run_id}")
                current_run_id = run_id
            else:
                error_msg = (
                    "No experiment run_id found. The API requires a valid experiment run_id to load model parameters.\n"
                    "Please either:\n"
                    "  1. Set RUN_ID environment variable to an existing experiment ID, OR\n"
                    "  2. Run dev_interactive_run.py first to create an experiment\n"
                    "Available experiments should be in the 'PROJECT_ROOT/experiments/' directory."
                    "Check your .env file to see your exact PROJECT_ROOT."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        # Load experiment configuration to get all parameters
        try:
            config = load_experiment_config(run_id)
            if not config:
                raise ValueError(f"Could not load config for run_id: {run_id}")
            
            # Extract parameters from experiment config - fail fast if missing
            try:
                configured_prompt_version = config['llm']['prompt_version']
                embedding_model = config['embedding']['model']
                llm_model = config['llm']['model']
                llm_temperature = config['llm']['temperature']
                llm_max_tokens = config['llm']['max_tokens']
            except KeyError as e:
                raise ValueError(f"Missing required parameter in experiment config: {e}")
            
            logger.info(f"Loaded experiment config for run_id: {run_id}")
            logger.info(f"Using prompt version: {configured_prompt_version}")
            logger.info(f"Using embedding model: {embedding_model}")
            logger.info(f"Using LLM model: {llm_model}")
            
        except Exception as e:
            logger.error(f"Failed to load experiment config: {e}")
            raise ValueError(f"Failed to load experiment config for run_id {run_id}: {e}")
        
        
        # Get Pinecone configuration from experiment config
        try:
            pinecone_index_name = config['retrieval']['index_name']
            pinecone_environment = config['retrieval']['environment']
        except KeyError as e:
            raise ValueError(f"Missing Pinecone configuration in experiment config: {e}")
        
        if not pinecone_index_name:
            raise ValueError("Pinecone index name is empty in experiment configuration")
        
        # Initialize retriever
        retriever = RAGRetriever(run_id=run_id, embedding_model=embedding_model, pinecone_index_name=pinecone_index_name)
        logger.info("RAG retriever initialized")
        
        # Initialize generator
        generator = RAGGenerator(
            run_id=run_id, 
            model=llm_model,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens
        )
        logger.info("RAG generator initialized")
        
        logger.info("RAG components initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG components: {e}")
        raise
    
    yield  # Application runs here
    
    # Shutdown code
    logger.info("Shutting down RAG API...")

def log_api_error(run_id: str, request_data: dict, error: Exception) -> None:
    """Log API question errors to file."""
    try:
        from config import construct_path
        logs_dir = construct_path('logs_dir', run_id)
        error_log_path = os.path.join(logs_dir, 'api_Qs_errors.log')
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        error_entry = f"\n{'='*60}\n"
        error_entry += f"Timestamp: {timestamp}\n"
        error_entry += f"Request JSON: {request_data}\n"
        error_entry += f"Error: {str(error)}\n"
        error_entry += f"{'='*60}\n"
        
        # Append to error log (create if doesn't exist)
        with open(error_log_path, 'a', encoding='utf-8') as f:
            f.write(error_entry)
        
        logger.info(f"Logged error to: {error_log_path}")
    except Exception as e:
        logger.error(f"Failed to log API error: {e}")

def save_api_response(run_id: str, question: str, response_data: dict, 
                      retrieved_chunks: list, quality_analysis: dict) -> None:
    """Save successful API response to file."""
    try:
        from config import construct_path
        answers_dir = construct_path('answers_to_testQ_dir', run_id)
        
        # Generate filename with timestamp and UUID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        filename = f"api_question_{timestamp}_{unique_id}.txt"
        filepath = os.path.join(answers_dir, filename)
        
        # Format chunks for output
        chunks_list = [f"{chunk['chunk_id']} (score: {chunk['score']:.3f})" 
                      for chunk in retrieved_chunks]
        
        # Create output content (same format as test queries)
        output_content = f"Question: {question}\n"
        output_content += f"Answer: {response_data['answer']}\n"
        output_content += f"chunks_retrieved: {chunks_list}\n"
        output_content += f"citations: {response_data['citations']}\n"
        output_content += f"\nResponse Quality Analysis:\n"
        for key, value in quality_analysis.items():
            output_content += f"  {key}: {value}\n"
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(output_content)
        
        logger.info(f"Saved API response to: {filepath}")
    except Exception as e:
        logger.error(f"Failed to save API response: {e}")

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

# Create FastAPI app
app = FastAPI(
    title="Typeform RAG Chatbot API",
    description="RAG-powered chatbot for Typeform Help Center",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # this is very permissive and should be tightened in production by specifying which domains' web pages can make browser requests to the API
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG components (initialized on startup)
retriever: Optional[RAGRetriever] = None
generator: Optional[RAGGenerator] = None
current_run_id: Optional[str] = None
configured_prompt_version: Optional[str] = None


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health check."""
    return HealthResponse(
        status="healthy",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        version="1.0.0"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        version="1.0.0"
    )

@app.post("/ask_question", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Main endpoint for asking questions to the RAG chatbot.
    
    Args:
        request: Question request with query
        
    Returns:
        QuestionResponse: Answer with sources and metadata
    """
    try:
        # Validate components are initialized
        if not retriever or not generator:
            raise HTTPException(
                status_code=503, 
                detail="RAG components not initialized"
            )
        
        # Validate request
        if not request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )
        
        logger.info(f"Processing question: '{request.question[:100]}...'")
        
        # Get top_k from experiment config - fail if missing
        try:
            config = load_experiment_config(current_run_id)
            top_k = config['retrieval']['top_k']
        except KeyError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Missing required parameter in experiment config: {e}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load experiment config: {e}"
            )
        
        # Retrieve context
        start_time = time.time()
        retrieved_chunks = retriever.retrieve_context(
            request.question, 
            top_k=top_k
        )
        
        # Generate response
        response_data = generator.generate_response(
            request.question,
            retrieved_chunks,
            prompt_version=configured_prompt_version
        )
        
        # Analyze response quality
        quality_analysis = analyze_response_quality(response_data)
        
        # Save successful response to file
        save_api_response(
            current_run_id, 
            request.question, 
            response_data, 
            retrieved_chunks,
            quality_analysis
        )
        
        # Calculate total processing time
        total_time = time.time() - start_time
        
        # Format sources
        sources = []
        for chunk in retrieved_chunks:
            source = SourceInfo(
                chunk_id=chunk.get('chunk_id', ''),
                doc_title=chunk.get('doc_title', ''),
                score=chunk.get('score', 0.0)
            )
            sources.append(source)
        
        # Prepare response
        response = QuestionResponse(
            question=request.question,
            answer=response_data['answer'],
            sources=sources,
            citations=response_data.get('citations', []),
            metadata={
                'generation_time_seconds': response_data.get('generation_time_seconds', 0.0),
                'total_time_seconds': total_time,
                'model_used': response_data.get('model_used', ''),
                'num_chunks_used': len(retrieved_chunks),
                'tokens_used': response_data.get('tokens_used', 0),
                'prompt_version': configured_prompt_version,
                'citation_coverage': response_data.get('citation_validation', {}).get('citation_coverage', 0.0)
            }
        )
        
        logger.info(f"Question processed successfully in {total_time:.2f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        # Log error to file
        request_data = {"question": request.question}
        log_api_error(current_run_id, request_data, e)
        
        logger.error(f"Error processing question: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/models")
async def get_models():
    """
    Get information about available models and configurations from the current experiment.
    
    Returns:
        dict: Information about available models and experiment configuration
    """
    try:
        # Get current run_id
        run_id = current_run_id
        if not run_id:
            raise HTTPException(
                status_code=503,
                detail="No experiment run_id available"
            )
        
        # Load experiment configuration
        try:
            config = load_experiment_config(run_id)
            if not config:
                raise HTTPException(
                    status_code=404,
                    detail=f"Configuration not found for run_id: {run_id}"
                )
        except Exception as e:
            logger.error(f"Error loading experiment config: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error loading experiment configuration: {str(e)}"
            )
        
        # Extract configuration sections
        experiment_info = config.get('experiment_info', {})
        embedding_config = config.get('embedding', {})
        llm_config = config.get('llm', {})
        retrieval_config = config.get('retrieval', {})
        chunking_config = config.get('chunking', {})
        
        return {
            "experiment_info": {
                "run_id": experiment_info.get('run_id', run_id),
                "created_at": experiment_info.get('created_at', 'unknown'),
                "description": experiment_info.get('description', 'RAG prototype experiment')
            },
            "embedding": {
                "model": embedding_config.get('model', 'unknown'),
                "dimension": embedding_config.get('dimension', 'unknown'),
                "batch_size": embedding_config.get('batch_size', 'unknown')
            },
            "llm": {
                "model": llm_config.get('model', 'unknown'),
                "temperature": llm_config.get('temperature', 'unknown'),
                "max_tokens": llm_config.get('max_tokens', 'unknown')
            },
            "retrieval": {
                "top_k": retrieval_config.get('top_k', 'unknown'),
                "index_name": retrieval_config.get('index_name', 'unknown')
            },
            "chunking": {
                "strategy": chunking_config.get('strategy', 'unknown'),
                "chunk_size": chunking_config.get('chunk_size', 'unknown'),
                "chunk_overlap": chunking_config.get('chunk_overlap', 'unknown'),
                "separators": chunking_config.get('separators', [])
            },
            "configured_prompt_version": configured_prompt_version
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving models: {str(e)}"
        )

# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return {
        "error": "Not Found",
        "message": "The requested endpoint was not found",
        "available_endpoints": [
            "GET /",
            "GET /health", 
            "POST /ask_question",
            "GET /models"
        ]
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {exc}")
    return {
        "error": "Internal Server Error",
        "message": "An unexpected error occurred"
    }

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
