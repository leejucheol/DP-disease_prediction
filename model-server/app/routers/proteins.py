from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from ..models.protein import Protein
from ..models.prediction import PredictionResult
from app.databases.protein import SequenceInput, DiseasePrediction, DiseasePredictionResponse
from app.databases.database_connect import get_db
from app.models.gcn_v0_1_0 import (
    model, esm_model, batch_converter, mlb, device, predict_top5_diseases
)
import uuid

router = APIRouter(prefix="/proteins", tags=["proteins"])

@router.post("/predict", response_model=DiseasePredictionResponse)
async def predict_disease(
    protein: SequenceInput,
    db: AsyncSession = Depends(get_db)
):
    try:
        sequence_id = str(uuid.uuid4())
        db_protein = Protein(
            sequence_id=sequence_id,
            sequence=protein.sequence,
            gene_id=None
        )
        db.add(db_protein)
        await db.commit()
        await db.refresh(db_protein)
        
        top5 = predict_top5_diseases(
            protein.sequence,
            model, esm_model,
            batch_converter, mlb, device
        )
        
        for pred in top5:
            db_pred = PredictionResult(
                sequence=db_protein.sequence,
                predicted_disease_id=pred["disease_id"],
                confidence_score=pred["probability"]
            )
            db.add(db_pred)
        await db.commit()
        
        predictions = [
            DiseasePrediction(
                disease_id=pred["disease_id"],
                disease_name=pred["disease_name"],
                probability=pred["probability"]
            ) for pred in top5
        ]
        
        return DiseasePredictionResponse(
            protein=db_protein.sequence,
            predictions=predictions
        )
        
    except Exception as e:
        await db.rollback()
        print(e)
        raise HTTPException(
            status_code=500,
            detail=f"예측 중 오류 발생: {str(e)}",
        )
    