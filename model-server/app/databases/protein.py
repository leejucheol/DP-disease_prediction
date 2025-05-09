from pydantic import BaseModel
from typing import List, Optional

class SequenceInput(BaseModel):
    sequence: str

class PredictionResponse(BaseModel):
    disease_id: str
    disease_name: str

class ProteinPredictionResponse(BaseModel):
    sequence: str
    predictions: List[PredictionResponse]
