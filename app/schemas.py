from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., example="Dis product make me happy well well")


class PredictResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
