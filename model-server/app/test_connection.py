import sys
import os

# 프로젝트 루트를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from app.database import engine

from sqlalchemy import text

async def test_connection():
    try:
        # 데이터베이스 연결 테스트
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
            print("✅ Database connection successful!")

            # 테이블 목록 조회
            result = await connection.execute(text("SHOW TABLES"))
            tables = result.fetchall()
            if tables:
                print("📋 Tables in the database:")
                for table in tables:
                    print(f"- {table[0]}")
            else:
                print("⚠️ No tables found in the database.")
    except Exception as e:
        print("❌ Database connection failed!")
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_connection())