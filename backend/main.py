

from backend.llm import translate_signs

from backend.schemas import (
    TranslateRequest,
    TranslateResponse
)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.config import (
    PROJECT_NAME,
    VERSION,
    ALLOWED_ORIGINS,
    MOCK_MODE,
    SIGN_LANGUAGE
)

from backend.schemas import (
    PredictRequest,
    PredictResponse
)

from backend.pipeline import run_prediction


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


@app.post(
    "/api/predict",
    response_model=PredictResponse
)
def predict(request: PredictRequest):

    try:
        return run_prediction(request.frames)

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except NotImplementedError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error)
        )

@app.post(
    "/api/translate",
    response_model=TranslateResponse
)
def translate(request: TranslateRequest):

    try:
        sentence = translate_signs(
            request.signs,
            request.language
        )

        return {
            "sentence": sentence
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error)
        )
