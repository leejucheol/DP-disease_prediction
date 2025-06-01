from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from app.schema.models import Protein
from app.schema.models import PredictDisease, ModelPredictionRun, PredictProtein
from app.schema.sequence_input_dto import SequenceInput, DiseasePrediction, DiseasePredictionResponse, DiseaseResponseProtein
from app.databases.database_connect import get_db
from app.repositories.prediction_dao import get_proteins_by_disease_dao
from app.models.gat_v0_1_0 import (
    model, esm_model, batch_converter, mlb, device, predict_top5_diseases
)
import uuid
from datetime import date

router = APIRouter(prefix="/proteins", tags=["proteins"])

@router.post("/predict", response_model=DiseasePredictionResponse)
async def predict_disease(
    protein: SequenceInput,
    db: AsyncSession = Depends(get_db)
):
    try:
        # 유니크한 ID 생성 (UUID)
        uniprot_id = str(uuid.uuid4())
        
        # 입력된 단백질 시퀀스를 Protein 테이블에 저장
        # gene_id 필드 제거
        db_protein = Protein(
            uniprot_id=uniprot_id,
            sequence=protein.sequence
        )
        db.add(db_protein)
        
        # 모델로 질병 예측 수행
        top5 = predict_top5_diseases(
            protein.sequence,
            model, esm_model,
            batch_converter, mlb, device
        )
        print(top5)
        
        # 모델 예측 실행 정보 저장
        model_run = ModelPredictionRun(
            input_sequence=protein.sequence,
            created_at=date.today(),
            model_version="gat_v0_1_0"  # 현재 사용 중인 모델 버전
        )
        db.add(model_run)
        await db.flush()  # run_id를 얻기 위해 flush
        
        # 예측된 단백질 정보 저장
        predict_protein = PredictProtein(
            run_id=model_run.run_id,
            pp_order=1,  # 입력 단백질은 항상 순위 1
            sequence=protein.sequence
        )
        db.add(predict_protein)
        
        # 예측된 질병 정보 저장
        for i, pred in enumerate(top5):
            predict_disease = PredictDisease(
                run_id=model_run.run_id,
                disease_name=pred["disease_name"],
                rank=i + 1  # 1부터 5까지의 순위
            )
            db.add(predict_disease)
        
        # 변경사항 커밋
        await db.commit()
        await db.refresh(db_protein)
        
        # 응답 구성
        predictions = [
            DiseasePrediction(
                disease_id=pred["disease_id"],
                disease_name=pred["disease_name"],
                probability=pred["probability"]
            ) for pred in top5
        ]
        
        return DiseasePredictionResponse(
            sequence=db_protein.sequence,  # 'protein'에서 'sequence'로 변경
            predictions=predictions
        )
        
    except Exception as e:
        await db.rollback()
        print(e)
        raise HTTPException(
            status_code=500,
            detail=f"예측 중 오류 발생: {str(e)}",
        )

@router.get("/diseases/{disease_id}", response_model=List[DiseaseResponseProtein])
async def get_proteins_by_disease(
    disease_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    특정 질병과 관련된 모든 단백질 서열 조회
    """
    try:
        proteins = await get_proteins_by_disease_dao(db, disease_id)
        if not proteins:
            raise HTTPException(status_code=404, detail="No proteins found for this disease")
        return proteins
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")