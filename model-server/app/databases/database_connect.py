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

# 5. 테이블 생성 함수 -------------------------------------------------------
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session 

async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

if __name__ == "__main__":
    import asyncio
    asyncio.run(create_tables())
