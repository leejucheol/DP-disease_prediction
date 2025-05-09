from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.databases.database_connect import Base, engine
from app.routers import proteins
from app.models.protein import Protein, Disease, GO_term, PDBStructure, ProteinAlias, PredictionResult

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield

app = FastAPI(lifespan=lifespan)
app.include_router(proteins.router)

@app.get("/")
async def root():
    return {"message": "데이터베이스 연결 성공!"}