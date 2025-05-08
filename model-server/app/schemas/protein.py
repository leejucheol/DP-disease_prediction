from pydantic import BaseModel
from typing import List, Optional

class ProteinBase(BaseModel):
    sequence: str
    uniprot_id: Optional[str] = None

class PredictionResponse(BaseModel):
    disease_id: str
    disease_name: str

class ProteinPredictionResponse(BaseModel):
    sequence: str
    predictions: List[PredictionResponse]
