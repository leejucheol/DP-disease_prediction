from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.models.protein import Protein

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