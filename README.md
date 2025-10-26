# Typeform RAG Prototype

A Retrieval-Augmented Generation (RAG) chatbot prototype for Typeform Help Center, built with FastAPI, Pinecone, and Google's Vertex AI.

## 🚀 Quick Start Options

After setting up your `.env` file (see Prerequisites Setup below), choose your preferred interaction method:

**Option A: Docker**
- Build and run with Docker - automatically creates experiment and starts API
- Access via web UI at http://localhost:8000/docs

**Option B: Interactive Development**
- Run `python dev_interactive_run.py` to create the first experiment and test queries
- All answers saved in `experiments/TIMESTAMP_RUNID/answers_to_testQ/`

**Option C: Local API Server**
- Run `python dev_interactive_run.py` first to create experiment
- Then run `python fastapi_serve.py` to start API server
- Access via web UI at http://localhost:8000/docs

---

## 📋 Prerequisites Setup

### Required Services
- **Google Cloud Project** with Vertex AI enabled
- **Pinecone account** and API key
- **Docker** (for Docker option) or **Python 3.11** (for local development)

### 1. Clone and Setup
```bash
git clone https://github.com/monica-dulgheru/typeform-rag-challenge.git
cd typeform-rag-challenge
```

### 2. Configure Credentials
```bash
# Create credentials directory
mkdir -p credentials

# Place your files in credentials/:
# - Google Cloud service account JSON file
# - Pinecone API key file (any name/extension)
```

### 3. Configure Environment
```bash
# Copy the example environment file
cp env.example .env

# Edit .env with your specific values:
# - PROJECT_ROOT: /app (for Docker) or /full/path/to/project (for local)
# - GOOGLE_CREDENTIALS_FILE: your-gcp-credentials.json
# - GCP_PROJECT_ID: your-gcp-project-id
# - PINECONE_API_KEY_FILE: your-pinecone-api-key-file
```

## 🐳 Method 1: Docker Container

### Docker Setup
```bash
# Build the Docker image
docker build -t typeform-rag-test .

# Run with volume mounts for credentials
docker run -d --name typeform-rag -p 8000:8000 \
  -v $(pwd)/credentials:/app/credentials:ro \
  --env-file .env \
  typeform-rag-test
```

### Two-Stage Startup Process
The container automatically handles two stages:

**Stage 1: Experiment Creation** (first run only)
- Runs `dev_interactive_run.py` to create embeddings and vector store
- Creates experiment directory with configuration
- Only runs if no experiments exist

**Stage 2: API Server**
- Starts FastAPI server using the experiment configuration
- Server available at http://localhost:8000

### Container Management
```bash
# Check container status
docker ps

# View logs (both stages)
docker logs -f typeform-rag

# Stop container
docker stop typeform-rag

# Remove container (next run will create new experiment)
docker rm typeform-rag
```

## 💻 Method 2: Local Development

### 1. Environment Setup
```bash
# Create virtual environment
python3.11 -m venv venv_type
source venv_type/bin/activate

# Install system dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev build-essential gcc g++ libxml2-dev libxslt1-dev libffi-dev libssl-dev curl wget git pkg-config

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Create Experiment
```bash
# Run the complete pipeline to create your first experiment
python dev_interactive_run.py
```

### 3. Start API Server
```bash
# Basic usage (auto-detects latest experiment)
python fastapi_serve.py

# With specific experiment
python fastapi_serve.py --run-id 20250118_143022_UTC_a1b2c3d4

# With custom host/port
python fastapi_serve.py --host 0.0.0.0 --port 8080

# With auto-reload for development
python fastapi_serve.py --reload
```

## 🔌 API Access

### Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint with health check |
| `/health` | GET | Health check endpoint |
| `/ask_question` | POST | Main RAG endpoint for asking questions |
| `/models` | GET | Get experiment configuration and model info |

### Using the API

**Option 1: Swagger UI (Recommended)**
1. Open http://localhost:8000/docs in your browser
2. Click on the `/ask_question` endpoint
3. Click "Try it out"
4. Enter your question in the request body:
   ```json
   {
     "question": "How do I create multi-language forms?"
   }
   ```
5. Click "Execute"

**Option 2: curl**
```bash
# Ask a question
curl -X POST 'http://localhost:8000/ask_question' \
     -H 'Content-Type: application/json' \
     -d '{"question": "How do I create multi-language forms?"}'

# Check health
curl http://localhost:8000/health

# Get model configuration
curl http://localhost:8000/models
```

---

## 📁 Project Structure

### Initial Structure (After Clone)
```
typeform-rag-challenge/
├── src/                               # Core RAG pipeline components
│   ├── knowledge_acquisition/         # HTML parsing and text cleaning
│   ├── chunking/                      # Document chunking pipeline
│   ├── embedding/                     # Embedding creation and Pinecone upload
│   ├── rag/                          # Query embedding, retrieval, and response generation
│   ├── api/                          # FastAPI application
│   └── utils.py                      # Utility functions
├── prompts/                          # Prompt templates and versions
├── knowledgebase/
│   └── raw_docs/                     # Raw HTML files (provided)
├── credentials/                      # API keys and credentials (user-provided)
├── config.py                         # Configuration management (loads from .env)
├── dev_interactive_run.py            # Interactive pipeline runner
├── fastapi_serve.py                  # API server launcher
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Container configuration
├── docker-compose.yml                # Multi-container setup
├── docker-entrypoint.sh
└── README.md                         # This file
└── README-discussion.md              # discussion of code/architecture/model choices and future directions
```

### After Running dev_interactive_run.py
```
typeform-rag-challenge/
├── [all files above, plus:]
├── knowledgebase/
│   ├── raw_docs/                     # Raw HTML files (provided)
│   └── clean_docs/                   # Cleaned documents (created on first run)
│       ├── docs/                     # Processed JSON documents
│       └── summary/                  # Document statistics
└── experiments/                      # Experiment runs with tracking
    └── 20250118_143022_UTC_a1b2c3d4/ # Timestamped experiment directory
        ├── chunks/                   # Document chunks
        │   ├── artifacts/            # chunks.json with all chunk data
        │   └── summary/              # chunk_stats.csv
        ├── vector-store-data/        # Embedding metadata
        ├── logs/                     # Execution logs and traces
        ├── answers_to_testQ/         # Test query results
        └── config.yaml               # Experiment configuration
```

---
## ⚙️ Configuration

The system separates **infrastructure** from **experiment** configuration:

### 1. Infrastructure Configuration (`.env` file)
Contains credentials and environment-specific settings. **Edit this once per environment.**

**Required Settings:**
- `PROJECT_ROOT`: `/app` (Docker) or `/full/path/to/project` (local)
- `GCP_PROJECT_ID`: Your Google Cloud Project ID
- `GOOGLE_CREDENTIALS_FILE`: Filename of your GCP credentials
- `PINECONE_API_KEY_FILE`: Filename of your Pinecone API key

### 2. Experiment Configuration (`dev_interactive_run.py`)
Contains model and RAG parameters. **Edit this to run different experiments.**

**Model Parameters (lines 97-116):**
- `EMBEDDING_MODEL`: `"text-embedding-005"`
- `EMBEDDING_DIMENSION`: `768`
- `LLM_MODEL`: `"gemini/gemini-2.5-flash"`
- `LLM_TEMPERATURE`: `0.1`
- `LLM_MAX_TOKENS`: `1024`
- `CHUNK_SIZE`: `512`
- `CHUNK_OVERLAP`: `100`
- `TOP_K`: `10`
- `PROMPT_VERSION`: `"detailed"` (options: `"basic"`, `"detailed"`, `"min"`)
- `PINECONE_INDEX_NAME`: `"typeform-help-rag"`
- `PINECONE_ENVIRONMENT`: `"us-east-1"`

### 3. Available Prompt Versions
- `"basic"`: Standard prompt with basic context formatting
- `"detailed"`: Enhanced prompt with metadata and detailed instructions
- `"min"`: Minimal prompt

---

## 🐛 Troubleshooting

### Common Issues

#### 1. "No experiments found" Error
```bash
# Docker: Check container logs
docker logs typeform-rag

# Local: Run experiment creation first
python dev_interactive_run.py
```

#### 2. Credentials Not Found
```bash
# Check credentials directory
ls -la credentials/
# Should show: your-gcp-credentials.json and your-pinecone-api-key-file

# For Docker, verify volume mount
docker exec typeform-rag ls -la /app/credentials/
```

#### 3. Google Cloud Authentication Error
```bash
# Verify credentials file exists and is accessible
ls -la credentials/your-gcp-credentials.json

# Check GCP_PROJECT_ID in .env file
cat .env | grep GCP_PROJECT_ID

# For Docker, verify file is mounted correctly
docker exec typeform-rag cat /app/credentials/your-gcp-credentials.json | head -5
```

#### 4. Pinecone Index Not Found
```bash
# Check Pinecone console for index existence
# Or modify PINECONE_INDEX_NAME in dev_interactive_run.py
```

#### 5. 'Cannot find SOME-DIR' 
This is likely an issue with the PROJECT_ROOT path
```bash
# For Docker: Should be exactly "/app" (no spaces)
PROJECT_ROOT=/app

# For local: Use full absolute path
PROJECT_ROOT=/home/user/typeform-rag-challenge
```

#### 6. Container Management
```bash
# Check container status
docker ps

# View all logs (both stages)
docker logs -f typeform-rag

# Restart container
docker restart typeform-rag

# Clean restart (removes experiment data)
docker stop typeform-rag
docker rm typeform-rag
# Then run docker run command again
```

#### 7. API Server Issues
```bash
# Check if server is running
curl http://localhost:8000/health

# View server logs
docker logs typeform-rag | grep -i "server\|api"

# Test specific endpoint
curl -X POST 'http://localhost:8000/ask_question' \
     -H 'Content-Type: application/json' \
     -d '{"question": "test"}'
```


