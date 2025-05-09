import os
from datetime import datetime
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship, declared_attr
from sqlalchemy import Column, String, Text, Integer, Float, ForeignKey, TIMESTAMP, Table
from dotenv import load_dotenv

# 1. 기본 베이스 클래스 설정 -------------------------------------------------
Base = declarative_base()

# 2. 데이터베이스 연결 설정 -------------------------------------------------
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

print("DATABASE_URL:", DATABASE_URL)
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL 환경변수가 비어 있습니다.")

engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

# 3. 모델 클래스 정의 -------------------------------------------------------

class Protein(Base):
    __tablename__ = "protein"
    
    sequence_id = Column(String(100), primary_key=True)
    sequence = Column(Text)
    gene_id = Column(Integer)
    
    # 관계 정의
    diseases = relationship("Disease", secondary="protein_disease")
    interactions = relationship(
        "ProteinInteraction",
        foreign_keys="[ProteinInteraction.sequence_id_1]"
    )
    go_terms = relationship("GO_term", secondary="protein_go")
    pdb_structures = relationship("PDBStructure", secondary="protein_pdb")
    aliases = relationship("ProteinAlias", back_populates="protein")

# 2. 질병 테이블
class Disease(Base):
    __tablename__ = "disease"
    
    disease_id = Column(String(50), primary_key=True)
    name = Column(String(255))
    proteins = relationship("Protein", secondary="protein_disease")

# 3. GO 용어 테이블
class GO_term(Base):
    __tablename__ = "go_term"
    term_id = Column(String(50), primary_key=True)

# 4. PDB 구조 테이블
class PDBStructure(Base):
    __tablename__ = "pdb_structure"
    pdb_id = Column(String(20), primary_key=True)

# 5. 단백질 별칭 테이블
class ProteinAlias(Base):
    __tablename__ = "protein_alias"
    alias_id = Column(Integer, primary_key=True, autoincrement=True)
    sequence_id = Column(String(100), ForeignKey("protein.sequence_id"))
    alias = Column(Text)
    
    protein = relationship("Protein", back_populates="aliases")

# 6. 예측 결과 테이블
class PredictionResult(Base):
    __tablename__ = "prediction_result"
    prediction_id = Column(Integer, primary_key=True, autoincrement=True)
    sequence = Column(Text)
    predicted_disease_id = Column(String(50), ForeignKey("disease.disease_id"))
    confidence_score = Column(Float)
    predicted_at = Column(TIMESTAMP, default=datetime.utcnow)
    
    disease = relationship("Disease")

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

# 5. 세션 관리 유틸리티 -----------------------------------------------------

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

# 6. 테이블 생성 함수 -------------------------------------------------------

async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

if __name__ == "__main__":
    import asyncio
    asyncio.run(create_tables())
