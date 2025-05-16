from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.schema.models import Protein, Disease, ProteinDisease

async def get_protein_by_id(db: AsyncSession, protein_id: str):
    """
    단백질 ID로 단백질 데이터를 조회
    """
    result = await db.execute(select(Protein).where(Protein.id == protein_id))
    return result.scalars().first()

async def get_proteins(db: AsyncSession, page: int, size: int, search: str = None):
    """
    단백질 목록 조회 (페이징 및 검색)
    """
    query = select(Protein)
    if search:
        query = query.where(Protein.name.ilike(f"%{search}%"))
    result = await db.execute(query.offset((page - 1) * size).limit(size))
    return result.scalars().all()   

async def get_proteins_by_disease_dao(db: AsyncSession, disease_id: str):
    """
    질병 아이디로 관련 단백질 서열 정보 조회
    """
    query = (
        select(Protein.sequence)
        .select_from(Protein)  
        .join(ProteinDisease, Protein.uniprot_id == ProteinDisease.uniprot_id)
        .where(Disease.disease_id == disease_id)
        .limit(5)
    )
    
    result = await db.execute(query)
    sequences = result.scalars().all()
    return [{"sequence": seq} for seq in sequences]