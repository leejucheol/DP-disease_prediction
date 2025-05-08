# main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.models.protein import Base
from app.database_connect import engine
from app.routers import proteins

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "데이터베이스 연결 성공!"}
