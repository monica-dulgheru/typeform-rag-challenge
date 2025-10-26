#!/bin/bash
set -e

echo "=========================================="
echo "Typeform RAG Docker Container Starting"
echo "=========================================="

# Check if experiments exist
if [ ! -d "experiments" ] || [ -z "$(ls -A experiments 2>/dev/null)" ]; then
    echo "No experiments found. Creating first experiment..."
    echo "This will create embeddings and set up the RAG system..."
    python dev_interactive_run.py
    echo "✓ First experiment created successfully!"
else
    echo "✓ Existing experiments found. Skipping experiment creation."
fi

echo "=========================================="
echo "Starting API server..."
echo "=========================================="

# Start the API server using fastapi_serve.py (handles experiment detection automatically)
python fastapi_serve.py --host 0.0.0.0 --port 8000
