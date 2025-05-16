import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.databases.database_connect import Base, engine
from app.routers import prediction_router
from app.databases.insert_basic_data import insert_basic_data

# 모든 모델 클래스 import
from app.schema.models import Protein, Disease, ProteinDisease
from app.schema.models import PredictProtein, ModelPredictionRun, PredictDisease

async def check_tables_exist():
    """데이터베이스에 필요한 테이블들이 이미 존재하는지 확인합니다."""
    required_tables = ['protein', 'disease', 'protein_disease', 
                        'predict_protein', 'model_prediction_run', 'predict_disease']
    
    async with engine.begin() as conn:
        # 데이터베이스에 존재하는 테이블 확인
        from sqlalchemy import text
        result = await conn.execute(text("SHOW TABLES"))
        existing_tables = [row[0] for row in result.fetchall()]
        
        # 모든 필요한 테이블이 존재하는지 확인
        for table in required_tables:
            if table not in existing_tables:
                return False
    
    return True

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. 데이터베이스 연결 확인
    print("데이터베이스 연결 확인 중...")
    
    try:
        # 2. 테이블 존재 확인
        tables_exist = await check_tables_exist()
        if not tables_exist:
            print("필요한 테이블이 존재하지 않습니다. 테이블을 생성합니다.")
            # SQLAlchemy ORM을 사용하여 테이블 생성
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            print("SQLAlchemy 모델을 사용하여 테이블 생성 완료")
        else:
            print("필요한 테이블이 모두 존재합니다.")
        
        # 3. 기본 데이터 삽입
        await insert_basic_data()
        
    except Exception as e:
        print(f"데이터베이스 초기화 중 오류 발생: {e}")
    
    yield

app = FastAPI(lifespan=lifespan)
app.include_router(prediction_router.router)

@app.get("/")
async def root():
    return {"message": "단백질 예측 프로그램 연결 성공"}