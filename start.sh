#!/bin/bash
# Start FastAPI REST API server in background
uvicorn api:app --host 0.0.0.0 --port 8889 &

# Start Streamlit frontend app
streamlit run app.py --server.port 8888 --server.address 0.0.0.0
