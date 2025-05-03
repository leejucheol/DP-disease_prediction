import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import as_declarative, declared_attr
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_async_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

@as_declarative()
class Base:
    id: int
    __name__: str

    @declared_attr
    def __tablename__(cls) -> str:
        return cls.__name__.lower()

async def get_db():
    async with SessionLocal() as session:
        yield session

import csv
from sqlalchemy import Table, Column, MetaData, String, Integer, Text, Float, ForeignKey, TIMESTAMP
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import insert

metadata = MetaData()

# 1. 단백질 테이블
protein_table = Table(
    "protein",
    metadata,
    Column("sequence_id", String(100), primary_key=True, comment="단백질 서열 ID"),
    Column("sequence", Text, comment="단백질 서열 (amino acids)"),
    Column("gene_id", Integer, comment="Gene ID"),
)

# 2. 질병 테이블
disease_table = Table(
    "disease",
    metadata,
    Column("disease_id", String(50), primary_key=True, comment="질병 ID"),
    Column("name", String(255), comment="질병 이름"),
)

# 3. 단백질-질병 매핑 테이블
protein_disease_table = Table(
    "protein_disease",
    metadata,
    Column("sequence_id", String(100), ForeignKey("protein.sequence_id"), comment="단백질 서열 ID"),
    Column("disease_id", String(50), ForeignKey("disease.disease_id"), comment="질병 ID"),
)

# 4. 단백질 상호작용 테이블
protein_interaction_table = Table(
    "protein_interaction",
    metadata,
    Column("sequence_id_1", String(100), ForeignKey("protein.sequence_id"), comment="상호작용 단백질 1"),
    Column("sequence_id_2", String(100), ForeignKey("protein.sequence_id"), comment="상호작용 단백질 2"),
    Column("combined_score", Integer, comment="상호작용 점수"),
)

# 5. GO 용어 테이블
go_term_table = Table(
    "go_term",
    metadata,
    Column("term_id", String(50), primary_key=True, comment="Gene Ontology ID"),
)

# 6. 단백질-GO 매핑
protein_go_table = Table(
    "protein_go",
    metadata,
    Column("sequence_id", String(100), ForeignKey("protein.sequence_id")),
    Column("term_id", String(50), ForeignKey("go_term.term_id")),
)

# 7. PDB 구조 테이블
pdb_structure_table = Table(
    "pdb_structure",
    metadata,
    Column("pdb_id", String(20), primary_key=True, comment="PDB 구조 ID"),
)

# 8. 단백질-PDB 매핑
protein_pdb_table = Table(
    "protein_pdb",
    metadata,
    Column("sequence_id", String(100), ForeignKey("protein.sequence_id")),
    Column("pdb_id", String(20), ForeignKey("pdb_structure.pdb_id")),
)

# 9. 단백질 별칭 테이블
protein_alias_table = Table(
    "protein_alias",
    metadata,
    Column("alias_id", Integer, primary_key=True, autoincrement=True),
    Column("sequence_id", String(100), ForeignKey("protein.sequence_id")),
    Column("alias", Text, comment="단백질 별칭"),
)

# 10. 예측 결과 테이블
prediction_result_table = Table(
    "prediction_result",
    metadata,
    Column("prediction_id", Integer, primary_key=True, autoincrement=True),
    Column("sequence", Text, comment="입력된 단백질 서열"),
    Column("predicted_disease_id", String(50), ForeignKey("disease.disease_id")),
    Column("confidence_score", Float),
    Column("predicted_at", TIMESTAMP, server_default="CURRENT_TIMESTAMP"),
)