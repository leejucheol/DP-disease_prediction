from pydantic import BaseModel
from typing import List

class SequenceInput(BaseModel):
    sequence: str

class DiseasePrediction(BaseModel):
    disease_id: str
    disease_name: str
    probability: float

class DiseasePredictionResponse(BaseModel):
    sequence: str  
    predictions: List[DiseasePrediction]
