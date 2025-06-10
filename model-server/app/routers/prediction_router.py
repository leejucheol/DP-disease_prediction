from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from app.schema.models import Protein
from app.schema.models import PredictDisease, ModelPredictionRun, PredictProtein
from app.schema.sequence_input_dto import SequenceInput, DiseasePrediction, DiseasePredictionResponse, DiseaseResponseProtein
from app.databases.database_connect import get_db
from app.repositories.prediction_dao import get_proteins_by_disease_dao
from app.models.gcn_v0_1_0 import (
    model as gcn_model,
    esm_model as gcn_esm_model,
    batch_converter as gcn_batch_converter,
    mlb as gcn_mlb,
    device as gcn_device,
    predict_top5_diseases as gcn_predict_top5_diseases
)
from app.models.gat_v0_1_0 import (
    model as gat_model,
    esm_model as gat_esm_model,
    batch_converter as gat_batch_converter,
    mlb as gat_mlb,
    device as gat_device,
    predict_top5_diseases as gat_predict_top5_diseases
)
import uuid
from datetime import date

router = APIRouter(prefix="/proteins", tags=["proteins"])

async def common_prediction(db: AsyncSession, sequence: str, predict_top5_diseases, model_version: str):
    try:
        # 유니크한 ID 생성 (UUID)
        uniprot_id = str(uuid.uuid4())
        
        # 입력된 단백질 시퀀스를 Protein 테이블에 저장
        # gene_id 필드 제거
        db_protein = Protein(
            uniprot_id=uniprot_id,
            sequence=sequence
        )
        db.add(db_protein)
        
        # 모델로 질병 예측 수행
        top5 = predict_top5_diseases(
            gcn_model if model_version == "gcn" else gat_model,
            gcn_esm_model if model_version == "gcn" else gat_esm_model,
            gcn_batch_converter if model_version == "gcn" else gat_batch_converter,
            gcn_mlb if model_version == "gcn" else gat_mlb,
            gcn_device if model_version == "gcn" else gat_device
        )
        print(top5)
        
        # 모델 예측 실행 정보 저장
        model_run = ModelPredictionRun(
            input_sequence=sequence,
            created_at=date.today(),
            model_version="gat_v0_1_0"  # 현재 사용 중인 모델 버전
        )
        db.add(model_run)
        await db.flush()  # run_id를 얻기 위해 flush
        
        # 예측된 단백질 정보 저장
        predict_protein = PredictProtein(
            run_id=model_run.run_id,
            pp_order=1,  # 입력 단백질은 항상 순위 1
            sequence=sequence
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
    
@router.post("/predict/gcn", response_model=DiseasePredictionResponse)
async def predict_disease_gcn(
    protein: SequenceInput,
    db: AsyncSession = Depends(get_db)
):
    try:
        return await common_prediction(
            db, 
            protein.sequence, 
            gcn_predict_top5_diseases, 
            "gcn"
        )
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"GCN 예측 중 오류 발생: {str(e)}",
        )

@router.post("/predict/gat", response_model=DiseasePredictionResponse)
async def predict_disease_gat(
    protein: SequenceInput,
    db: AsyncSession = Depends(get_db)
):
    try:
        return await common_prediction(
            db, 
            protein.sequence, 
            gat_predict_top5_diseases, 
            "gat"
        )
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"GAT 예측 중 오류 발생: {str(e)}",
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