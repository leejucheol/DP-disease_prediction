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
    Column("sequence_id", String(100), ForeignKey("protein.sequence_id"), primary_key=True),
    Column("disease_id", String(50), ForeignKey("disease.disease_id"), primary_key=True)
)

# 2. 단백질 상호작용 테이블
protein_interaction = Table(
    "protein_interaction",
    Base.metadata,
    Column("sequence_id_1", String(100), ForeignKey("protein.sequence_id"), primary_key=True),
    Column("sequence_id_2", String(100), ForeignKey("protein.sequence_id"), primary_key=True),
    Column("combined_score", Integer)
)

# 3. 단백질-GO 매핑
protein_go = Table(
    "protein_go",
    Base.metadata,
    Column("sequence_id", String(100), ForeignKey("protein.sequence_id"), primary_key=True),
    Column("term_id", String(50), ForeignKey("go_term.term_id"), primary_key=True)
)

# 4. 단백질-PDB 매핑
protein_pdb = Table(
    "protein_pdb",
    Base.metadata,
    Column("sequence_id", String(100), ForeignKey("protein.sequence_id"), primary_key=True),
    Column("pdb_id", String(20), ForeignKey("pdb_structure.pdb_id"), primary_key=True)
)

# 3. 세션 관리 유틸리티 -----------------------------------------------------

# 1. GO 용어 테이블
class GO_term(Base):
    __tablename__ = "go_term"
    term_id = Column(String(50), primary_key=True)

# 2. PDB 구조 테이블
class PDBStructure(Base):
    __tablename__ = "pdb_structure"
    pdb_id = Column(String(20), primary_key=True)

# 3. 단백질 별칭 테이블
class ProteinAlias(Base):
    __tablename__ = "protein_alias"
    alias_id = Column(Integer, primary_key=True, autoincrement=True)
    sequence_id = Column(String(100), ForeignKey("protein.sequence_id"))
    alias = Column(Text)
    
    protein = relationship("Protein", back_populates="aliases")

class Protein(Base):
    __tablename__ = "protein"
    sequence_id = Column(String(100), primary_key=True)
    sequence = Column(Text)
    gene_id = Column(Integer)
    diseases = relationship("Disease", secondary=protein_disease)
    go_terms = relationship("GO_term", secondary=protein_go)
    pdb_structures = relationship("PDBStructure", secondary=protein_pdb)
    aliases = relationship("ProteinAlias", back_populates="protein")