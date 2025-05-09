# prediction.py
from sqlalchemy import Column, Integer, String, Text, Float, TIMESTAMP, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from app.databases.database_connect import Base

# 4. 예측 결과 테이블
class PredictionResult(Base):
    __tablename__ = "prediction_result"
    prediction_id = Column(Integer, primary_key=True, autoincrement=True)
    sequence = Column(Text)
    predicted_disease_id = Column(String(50), ForeignKey("disease.disease_id"))
    confidence_score = Column(Float)
    predicted_at = Column(TIMESTAMP, default=datetime.utcnow)
    
    disease = relationship("Disease")
