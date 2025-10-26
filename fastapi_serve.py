#!/usr/bin/env python3
"""
Interactive Server Script for Typeform RAG Prototype

This script starts the FastAPI server for interactive user prompts,
completely separate from the development entrypoint.py file.

Usage:
    python fastapi_serve.py                           # Use latest experiment
    python fastapi_serve.py --run-id <run_id>         # Use specific experiment
    python fastapi_serve.py --host 0.0.0.0 --port 8080  # Custom host/port
    python fastapi_serve.py --reload                  # Enable auto-reload for development

Examples:
    # Start server with latest experiment
    python fastapi_serve.py
    
    # Start server with specific experiment
    python fastapi_serve.py --run-id 20251018_131827_UTC_0a79abf5
    
    # Start server on different port
    python fastapi_serve.py --port 8080
    
    # Start server with auto-reload (development)
    python fastapi_serve.py --reload
"""

import argparse
import sys
import os
import uvicorn
import logging
from typing import Optional

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()  # Loads .env file if it exists

# Import project modules
from config import get_latest_run_id, validate_run_id, load_experiment_config
from src.rag.retriever import RAGRetriever
from src.rag.generator import RAGGenerator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Start Typeform RAG FastAPI server for interactive prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Use latest experiment
  %(prog)s --run-id <run_id>         # Use specific experiment
  %(prog)s --host 0.0.0.0 --port 8080  # Custom host/port
  %(prog)s --reload                  # Enable auto-reload for development
        """
    )
    
    parser.add_argument(
        '--run-id',
        type=str,
        help='Experiment run ID to use (default: latest available)'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind the server to (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to bind the server to (default: 8000)'
    )
    
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Enable auto-reload for development (default: False)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='info',
        choices=['debug', 'info', 'warning', 'error'],
        help='Log level (default: info)'
    )
    
    return parser.parse_args()

def validate_and_get_run_id(run_id: Optional[str]) -> str:
    """Validate run_id and return the one to use."""
    if run_id is None:
        # Get latest run ID
        latest_run_id = get_latest_run_id()
        if latest_run_id is None:
            logger.error("No experiments found. Please run the pipeline first using entrypoint.py")
            sys.exit(1)
        logger.info(f"Using latest experiment: {latest_run_id}")
        return latest_run_id
    else:
        # Validate provided run ID
        if not validate_run_id(run_id):
            logger.error(f"Invalid run ID: {run_id}")
            logger.error("Run ID should be in format: YYYYMMDD_HHMMSS_TZ_<UUID>")
            sys.exit(1)
        logger.info(f"Using specified experiment: {run_id}")
        return run_id

def get_model_params_from_config(run_id: str) -> dict:
    """Load model parameters from experiment configuration."""
    try:
        config = load_experiment_config(run_id)
        
        embedding_config = config.get('embedding', {})
        llm_config = config.get('llm', {})
        retrieval_config = config.get('retrieval', {})
        
        params = {
            'embedding_model': embedding_config.get('model'),
            'embedding_dimension': embedding_config.get('dimension'),
            'llm_model': llm_config.get('model'),
            'llm_temperature': llm_config.get('temperature'),
            'llm_max_tokens': llm_config.get('max_tokens'),
            'top_k': retrieval_config.get('top_k'),
            'prompt_version': llm_config.get('prompt_version'),
            'pinecone_index_name': retrieval_config.get('index_name'),
            'pinecone_environment': retrieval_config.get('environment')
        }
        
        # Validate that all required parameters are present
        missing_params = [k for k, v in params.items() if v is None]
        if missing_params:
            raise ValueError(f"Missing required parameters in experiment config: {missing_params}")
        
        return params
        
    except Exception as e:
        logger.error(f"Could not load model parameters from experiment config: {e}")
        logger.error("Make sure the experiment was created with a valid configuration")
        sys.exit(1)

def get_prompt_version_from_config(run_id: str) -> str:
    """Get the prompt version used in the experiment configuration."""
    params = get_model_params_from_config(run_id)
    return params['prompt_version']

def test_rag_components(run_id: str, model_params: dict) -> bool:
    """Test that RAG components can be initialized with the given run_id and parameters."""
    try:
        logger.info("Testing RAG components initialization...")
        
        # Test retriever
        retriever = RAGRetriever(
            run_id=run_id, 
            embedding_model=model_params['embedding_model'],
            pinecone_index_name=model_params['pinecone_index_name']
        )
        logger.info("✓ RAG retriever initialized successfully")
        
        # Test generator
        generator = RAGGenerator(
            run_id=run_id, 
            model=model_params['llm_model'],
            temperature=model_params['llm_temperature'],
            max_tokens=model_params['llm_max_tokens']
        )
        logger.info("✓ RAG generator initialized successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to initialize RAG components: {e}")
        logger.error("Make sure the experiment has completed successfully and embeddings are available")
        return False

def print_server_info(host: str, port: int, run_id: str):
    """Print server startup information."""
    print("\n" + "="*60)
    print("🚀 TYPEFORM RAG SERVER STARTING")
    print("="*60)
    print(f"📊 Experiment Run ID: {run_id}")
    print(f"🌐 Server URL: http://{host}:{port}")
    print(f"📚 API Documentation: http://{host}:{port}/docs")
    print(f"❤️  Health Check: http://{host}:{port}/health")
    print("\n💡 How to use:")
    print("   1. Open http://localhost:8000/docs in your browser")
    print("   2. Click on 'POST /ask_question' endpoint")
    print("   3. Click 'Try it out' button")
    print("   4. Enter your question in the request body")
    print("   5. Click 'Execute' to get your answer")
    print("\n🔧 Test with curl:")
    print(f'   curl -X POST "http://localhost:{port}/ask_question" \\')
    print('        -H "Content-Type: application/json" \\')
    print('        -d \'{"question": "How do I create multi-language forms?"}\'')
    print("\n⏹️  Press Ctrl+C to stop the server")
    print("="*60 + "\n")

def main():
    """Main function to start the server."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Validate and get run ID
        run_id = validate_and_get_run_id(args.run_id)
        
        # Load model parameters from experiment config
        model_params = get_model_params_from_config(run_id)
        logger.info(f"Loaded model parameters from experiment config")
        
        # Test RAG components
        if not test_rag_components(run_id, model_params):
            sys.exit(1)
        
        # Get prompt version from model params
        prompt_version = model_params['prompt_version']
        logger.info(f"Using prompt version: {prompt_version}")
        
        # Print server information
        print_server_info(args.host, args.port, run_id)
        
        # Set environment variables for the FastAPI app to use
        os.environ['RUN_ID'] = run_id
        os.environ['RAG_PROMPT_VERSION'] = prompt_version
        os.environ['RAG_EMBEDDING_MODEL'] = model_params['embedding_model']
        os.environ['RAG_LLM_MODEL'] = model_params['llm_model']
        os.environ['RAG_LLM_TEMPERATURE'] = str(model_params['llm_temperature'])
        os.environ['RAG_LLM_MAX_TOKENS'] = str(model_params['llm_max_tokens'])
        os.environ['RAG_TOP_K'] = str(model_params['top_k'])
        
        # Start the server
        logger.info(f"Starting FastAPI server on {args.host}:{args.port}")
        
        uvicorn.run(
            "src.api.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level,
            access_log=True
        )
        
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
        logger.info("Server shutdown complete")
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
