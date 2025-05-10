import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from app.utils.csv_loader import insert_data_from_csv  
from app.databases.database_connect import AsyncSessionLocal

async def main():
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "models/data/processed_train_with_unchar.csv"))
    async with AsyncSessionLocal() as session:
        await insert_data_from_csv(csv_path, session)


asyncio.run(main())