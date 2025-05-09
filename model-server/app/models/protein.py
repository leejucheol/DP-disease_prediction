# app/models/protein.py
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Text
from app.databases.database_connect import Base

class Protein(Base):
    __tablename__ = "protein"
    
    sequence_id = Column(String(100), primary_key=True)
    sequence = Column(Text)
    gene_id = Column(Integer)
