from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from .. import schemas, models
from ..database import get_db

from dotenv import load_dotenv
import os

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

from app.models.gcn_v0_1_0 import (
    model, esm_model, batch_converter, mlb, device, predict_top5_diseases
)

router = APIRouter(
    prefix="/proteins",
    tags=["proteins"]
)

@router.post("/predict", response_model=schemas.PredictionResponse)
async def predict_protein(
    protein: schemas.ProteinCreate,
    db: AsyncSession = Depends(get_db)
):
    try:
        # 1. 단백질 DB 저장
        db_protein = models.Protein(**protein.dict())
        db.add(db_protein)
        await db.commit()
        await db.refresh(db_protein)
        
        # 2. 예측 실행
        top5 = await predict_top5_diseases(
            protein.sequence, 
            model, esm_model, 
            batch_converter, mlb, device
        )
        
        # 3. 예측 결과 DB 저장
        for pred in top5:
            db_pred = models.Prediction(
                protein_id=db_protein.id,
                disease_id=pred["disease_id"],
                probability=pred["probability"]  # 필드 추가
            )
            db.add(db_pred)
        await db.commit()
        
        return {
            "protein": db_protein,
            "predictions": top5
        }
        
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=500, 
            detail=f"예측 중 오류 발생: {str(e)}"
        )
