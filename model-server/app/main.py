import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.databases.database_connect import Base, engine
from app.routers import proteins

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield

app = FastAPI(lifespan=lifespan)
app.include_router(proteins.router)

@app.get("/")
async def root():
    return {"message": "단백질 예측 프로그램 연결 성공"}