import asyncio
from database import engine, insert_data_from_csv, SessionLocal

async def main():
    async with SessionLocal() as session:
        await insert_data_from_csv(
            "d:\\Users\\main\\folder\\workspace\\01_division_projects\\dvision_projects\\DP-disease_prediction\\model-server\\data\\processed_train_small.csv",
            session
        )

asyncio.run(main())