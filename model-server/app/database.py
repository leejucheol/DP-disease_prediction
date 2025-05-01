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
from sqlalchemy import Table, MetaData, Column, String, Integer, Text, ForeignKey
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import insert

# Define the tables based on the ERD
metadata = MetaData()

protein_table = Table(
    "protein",
    metadata,
    Column("uniprot_id", String(50), primary_key=True, comment="UniProt 식별자"),
    Column("gene_id", Integer, comment="Gene ID"),
    Column("sequence", Text, comment="아미노산 서열"),
)

disease_table = Table(
    "disease",
    metadata,
    Column("disease_id", String(50), primary_key=True, comment="질병 식별자"),
    Column("name", String(255), comment="질병 이름"),
)

protein_disease_table = Table(
    "protein_disease",
    metadata,
    Column("uniprot_id", String(50), ForeignKey("protein.uniprot_id"), comment="PROTEIN.uniprot_id"),
    Column("disease_id", String(50), ForeignKey("disease.disease_id"), comment="DISEASE.disease_id"),
)

protein_interaction_table = Table(
    "protein_interaction",
    metadata,
    Column("protein1", String(50), ForeignKey("protein.uniprot_id"), comment="PROTEIN.uniprot_id"),
    Column("protein2", String(50), ForeignKey("protein.uniprot_id"), comment="PROTEIN.uniprot_id"),
    Column("combined_score", Integer, comment="상호작용 점수"),
)

# Function to insert data from CSV
async def insert_data_from_csv(file_path: str, session: AsyncSession):
    """
    Reads data from a CSV file and inserts it into the database.
    """
    async with session.begin():
        with open(file_path, mode='r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                # Insert into protein table
                stmt_protein = insert(protein_table).values(
                    uniprot_id=row["UniProt_ID"],
                    gene_id=int(row["Gene ID"]),
                    sequence=row["sequence"],
                )
                await session.execute(stmt_protein)

                # Insert into disease table
                stmt_disease = insert(disease_table).values(
                    disease_id=row["Disease ID"],
                    name=row["Disease Name"],
                )
                await session.execute(stmt_disease)

                # Insert into protein_disease table
                stmt_protein_disease = insert(protein_disease_table).values(
                    uniprot_id=row["UniProt_ID"],
                    disease_id=row["Disease ID"],
                )
                await session.execute(stmt_protein_disease)

                # Insert into protein_interaction table
                stmt_protein_interaction = insert(protein_interaction_table).values(
                    protein1=row["protein1"],
                    protein2=row["protein2"],
                    combined_score=int(row["combined_score"]),
                )
                await session.execute(stmt_protein_interaction)

        print("Data inserted successfully!")