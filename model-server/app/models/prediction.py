# prediction.py
from sqlalchemy import Column, Integer, String, Date  # ✅ Date를 여기서 임포트
from sqlalchemy.sql.sqltypes import TIMESTAMP  # ✅ TIMESTAMP 추가
from sqlalchemy.orm import relationship
from datetime import datetime  # ⚠️ datetime.utcnow() 사용 시 필요
from app.databases.database_connect import Base

# 1. 예측 결과 테이블
class PredictDisease(Base):
    __tablename__ = "predict_disease"
    run_id = Column(Integer, primary_key=True)
    disease_name = Column(String(100), nullable=False)
    rank = Column(Integer, nullable=False)

# 모델 예측 로그 테이블
class ModelPredictionRun(Base):
    __tablename__ = "model_prediction_run"
    run_id = Column(Integer, primary_key=True)
    input_sequence = Column(String(1000))
    created_at = Column(Date, nullable=False)
    model_version = Column(String(20))

# 예측된 단백질 테이블
class PredictProtein(Base):
    __tablename__ = "predict_protein"
    run_id = Column(Integer, primary_key=True)
    rank = Column(Integer, nullable=False)
    sequence = Column(String(1000))