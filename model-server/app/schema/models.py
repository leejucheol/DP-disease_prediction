from sqlalchemy import Column, String, Text, Integer, Date, ForeignKey
from sqlalchemy.orm import relationship
from app.databases.database_connect import Base

class Protein(Base):
    __tablename__ = "protein"
    
    uniprot_id = Column(String(100), primary_key=True)
    sequence = Column(Text, nullable=False)
    
    # 관계 정의
    diseases = relationship("ProteinDisease", back_populates="protein")

class Disease(Base):
    __tablename__ = "disease"
    
    disease_id = Column(String(100), primary_key=True)
    disease_name = Column(String(100), nullable=False)
    
    # 관계 정의
    proteins = relationship("ProteinDisease", back_populates="disease")

class ProteinDisease(Base):
    __tablename__ = "protein_disease"
    
    uniprot_id = Column(String(100), ForeignKey("protein.uniprot_id"), primary_key=True)
    disease_id = Column(String(100), ForeignKey("disease.disease_id"), primary_key=True)
    
    # 관계 정의
    protein = relationship("Protein", back_populates="diseases")
    disease = relationship("Disease", back_populates="proteins")

class PredictProtein(Base):
    __tablename__ = "predict_protein"
    
    run_id = Column(Integer, primary_key=True)
    rank = Column(Integer, nullable=False)
    sequence = Column(Text, nullable=False)
    
    # 관계 정의
    predicted_diseases = relationship("PredictDisease", back_populates="predict_protein")

class ModelPredictionRun(Base):
    __tablename__ = "model_prediction_run"
    
    run_id = Column(Integer, primary_key=True)
    input_sequence = Column(Text, nullable=False)
    created_at = Column(Date, nullable=False)
    model_version = Column(String(20), nullable=False)

class PredictDisease(Base):
    __tablename__ = "predict_disease"
    
    run_id = Column(Integer, ForeignKey("predict_protein.run_id"), primary_key=True)
    disease_name = Column(String(100), nullable=False)
    rank = Column(Integer, primary_key=True)  # rank도 primary key로 설정
    
    # 관계 정의
    predict_protein = relationship("PredictProtein", back_populates="predicted_diseases")
