from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.schema.models import PredictDisease, ModelPredictionRun, InputProtein, Protein
from app.schema.sequence_input_dto import SequenceInput, DiseasePrediction, DiseasePredictionResponse
from app.databases.database_connect import get_db
from app.models.gcn_v0_1_0 import (
    model, esm_model, batch_converter, mlb, device, predict_top5_diseases
)
import uuid
from datetime import date
from sqlalchemy import select
from typing import List

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
            model_version="gcn_v0_1_0"  # 현재 사용 중인 모델 버전
        )
        db.add(model_run)
        await db.flush()  # run_id를 얻기 위해 flush
        
        # 예측된 단백질 정보 저장
        predict_protein = InputProtein(
            run_id=model_run.run_id,
            sequence=protein.sequence
        )
        db.add(predict_protein)
        
        # 예측된 질병 정보 저장
        for i, pred in enumerate(top5):
            predict_disease = PredictDisease(
                run_id=model_run.run_id,
                disease_name=pred["disease_name"],
                pd_rank=i + 1,  # 1부터 5까지의 순위
                probability=pred["probability"]
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
            sequence=db_protein.sequence,  
            predictions=predictions
        )
        
    except Exception as e:
        await db.rollback()
        print(e)
        raise HTTPException(
            status_code=500,
            detail=f"예측 중 오류 발생: {str(e)}",
        )

@router.get("/diseases/{disease_id}/proteins", response_model=List[Protein])
async def get_proteins_by_disease(
    disease_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    질병 ID로 연관된 단백질 목록 반환
    """
    # PredictDisease에서 해당 질병의 run_id 추출
    result = await db.execute(
        select(PredictDisease.run_id).where(PredictDisease.disease_name == disease_id)
    )
    run_ids = [row[0] for row in result.fetchall()]
    if not run_ids:
        return []

    # InputProtein에서 run_id로 단백질 시퀀스 조회
    result = await db.execute(
        select(InputProtein.sequence).where(InputProtein.run_id.in_(run_ids))
    )
    sequences = [row[0] for row in result.fetchall()]

    # Protein 테이블에서 시퀀스 정보 조회
    result = await db.execute(
        select(Protein).where(Protein.sequence.in_(sequences))
    )
    proteins = [row[0] for row in result.fetchall()]
    return proteins