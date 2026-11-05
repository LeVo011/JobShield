"""Application configuration and local paths."""
from pathlib import Path
import os
from dotenv import load_dotenv
ROOT=Path(__file__).resolve().parent.parent
load_dotenv(ROOT/".env")
DATA_DIR=ROOT/"data"; KB_DIR=DATA_DIR/"knowledge_base"; MODEL_DIR=ROOT/"models"; MODEL_PATH=MODEL_DIR/"classifier.joblib"; EVAL_PATH=DATA_DIR/"evaluation_results.json"
SEED=42; MAX_MESSAGE_CHARS=20000; MAX_UPLOAD_MB=10
EMBEDDING_MODEL=os.getenv("JOBSHIELD_EMBEDDING_MODEL","sentence-transformers/all-MiniLM-L6-v2")
LLM_API_KEY=os.getenv("JOBSHIELD_LLM_API_KEY",""); LLM_ENDPOINT=os.getenv("JOBSHIELD_LLM_ENDPOINT",""); LLM_MODEL=os.getenv("JOBSHIELD_LLM_MODEL","")
