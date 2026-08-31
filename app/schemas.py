from pydantic import BaseModel, Field


class SentimentRequest(BaseModel):
    text: str = Field(..., example="Dis product make me happy well well")


class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
