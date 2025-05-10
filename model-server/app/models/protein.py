# app/models/protein.py
from sqlalchemy.orm import relationship
from app.databases.database_connect import Base
from sqlalchemy import Column, Integer, String, Text, Table, Float, TIMESTAMP, ForeignKey
from datetime import datetime

# 4. 연결 테이블 정의 (Many-to-Many 관계) -----------------------------------
# 1. 단백질-질병 매핑 테이블
protein_disease = Table(
    "protein_disease",
    Base.metadata,
    Column("uniprot_id", String(100), ForeignKey("protein.uniprot_id"), primary_key=True),
    Column("disease_id", String(50), ForeignKey("disease.disease_id"), primary_key=True)
)

class Protein(Base):
    __tablename__ = "protein"
    uniprot_id = Column(String(100), primary_key=True)
    sequence = Column(Text)