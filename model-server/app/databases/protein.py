from pydantic import BaseModel

class SequenceInput(BaseModel):
    sequence: str

class DiseasePrediction(BaseModel):
    disease_id: str
    disease_name: str
    probability: float

class DiseasePredictionResponse(BaseModel):
    sequence: str
    predictions: list[DiseasePrediction]
