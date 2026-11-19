from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import (
    PROJECT_NAME,
    VERSION,
    ALLOWED_ORIGINS,
    MOCK_MODE,
    SIGN_LANGUAGE
)

app = FastAPI(
    title=PROJECT_NAME,
    version=VERSION,
    description="AI-powered sign language recognition"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/")
def root():
    return {
        "project": PROJECT_NAME,
        "version": VERSION,
        "status": "running"
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "mock_mode": MOCK_MODE,
        "sign_language": SIGN_LANGUAGE
    }