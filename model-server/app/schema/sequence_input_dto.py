from pydantic import BaseModel
from typing import List

class SequenceInput(BaseModel):
    sequence: str

class DiseasePrediction(BaseModel):
    disease_id: str
    disease_name: str
    probability: float

class DiseasePredictionResponse(BaseModel):
    sequence: str  # 'protein' 대신 'sequence'로 변경
    predictions: List[DiseasePrediction]
