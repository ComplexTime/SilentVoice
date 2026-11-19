from pydantic import BaseModel, Field
from typing import Literal

from backend.config import SEQUENCE_LENGTH


class PredictRequest(BaseModel):
    frames: list[str] = Field(
        min_length=SEQUENCE_LENGTH,
        max_length=SEQUENCE_LENGTH
    )


class PredictResponse(BaseModel):
    sign: str
    confidence: float = Field(ge=0.0, le=1.0)


class TranslateRequest(BaseModel):
    signs: list[str] = Field(
        min_length=1,
        max_length=50
    )
    language: Literal["en", "he", "ru"] = "en"


class TranslateResponse(BaseModel):
    sentence: str