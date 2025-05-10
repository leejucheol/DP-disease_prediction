# disease.py

from sqlalchemy import Column, String
from sqlalchemy.orm import relationship
from app.databases.database_connect import Base
from app.models.protein import protein_disease

# 2. 질병 테이블
class Disease(Base):
    __tablename__ = "disease"
    
    disease_id = Column(String(50), primary_key=True)
    disease_name = Column(String(255))