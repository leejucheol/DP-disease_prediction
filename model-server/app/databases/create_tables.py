"""
데이터베이스 테이블 생성 스크립트
"""
import os
from sqlalchemy import text
from app.databases.database_connect import engine

async def create_tables():
    """데이터베이스에 필요한 테이블들을 생성합니다."""
    
    # SQL 파일 경로 - 현재 파일과 같은 디렉토리에 있는 create_tables.sql 파일을 사용
    sql_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'create_tables.sql')
    
    print(f"SQL 파일 경로: {sql_file_path}")
    
    # SQL 파일 읽기
    with open(sql_file_path, 'r', encoding='utf-8') as file:
        sql_commands = file.read()
    
    # SQL 명령어 실행
    async with engine.begin() as conn:
        # 각 SQL 명령어를 하나씩 실행
        for command in sql_commands.split(';'):
            if command.strip():
                await conn.execute(text(command))
    
    print("데이터베이스 테이블 생성 완료")

async def check_tables_exist():
    """데이터베이스에 필요한 테이블들이 이미 존재하는지 확인합니다."""
    required_tables = ['protein', 'disease', 'protein_disease', 
                      'input_protein', 'model_prediction_run', 'predict_disease']
    
    async with engine.begin() as conn:
        # 데이터베이스에 존재하는 테이블 확인
        result = await conn.execute(text("SHOW TABLES"))
        existing_tables = [row[0] for row in result.fetchall()]
        
        # 모든 필요한 테이블이 존재하는지 확인
        for table in required_tables:
            if table not in existing_tables:
                return False
    
    return True
