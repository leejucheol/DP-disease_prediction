import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from app.utils.csv_loader import insert_data_from_csv  
from app.database_connect import SessionLocal

async def main():
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "models/data/processed_train_small.csv"))
    async with SessionLocal() as session:
        await insert_data_from_csv(csv_path, session)


asyncio.run(main())