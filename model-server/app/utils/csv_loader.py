import csv
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import insert
from app.databases.database_connect import (
    protein_table,
    disease_table,
    protein_disease_table,
    protein_interaction_table,
    go_term_table,
    protein_go_table,
    pdb_structure_table,
    protein_pdb_table,
    protein_alias_table,
)

def split_terms(raw, delimiter=';'):
    if not raw or not isinstance(raw, str):
        return []
    return [t.strip() for t in raw.split(delimiter) if t.strip()]

async def insert_data_from_csv(file_path: str, session: AsyncSession):
    async with session.begin():
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                try:
                    sequence_id = row["UniProt_ID"]
                    gene_id = int(row["Gene ID"]) if row["Gene ID"].isdigit() else None
                    sequence = row["sequence"]
                    disease_id = row["Disease ID"]
                    disease_name = row["Disease Name"]

                    # 1. 단백질 정보
                    await session.execute(insert(protein_table).values(
                        sequence_id=sequence_id,
                        gene_id=gene_id,
                        sequence=sequence,
                    ))

                    # 2. 질병 정보
                    await session.execute(insert(disease_table).values(
                        disease_id=disease_id,
                        name=disease_name,
                    ))

                    # 3. 단백질–질병 관계
                    await session.execute(insert(protein_disease_table).values(
                        sequence_id=sequence_id,
                        disease_id=disease_id,
                    ))

                    # 4. 상호작용
                    await session.execute(insert(protein_interaction_table).values(
                        sequence_id_1=row["protein1"],
                        sequence_id_2=row["protein2"],
                        combined_score=int(row["combined_score"]),
                    ))

                    # 5. GO Term
                    for term in split_terms(row.get("GO_Terms")):
                        await session.execute(insert(go_term_table).values(term_id=term))
                        await session.execute(insert(protein_go_table).values(
                            sequence_id=sequence_id,
                            term_id=term,
                        ))

                    # 6. PDB 구조
                    for pdb_id in split_terms(row.get("PDB_IDs")):
                        await session.execute(insert(pdb_structure_table).values(pdb_id=pdb_id))
                        await session.execute(insert(protein_pdb_table).values(
                            sequence_id=sequence_id,
                            pdb_id=pdb_id,
                        ))

                    # 7. 단백질 별칭
                    alias = row.get("proteinFullNames")
                    if alias and isinstance(alias, str):
                        await session.execute(insert(protein_alias_table).values(
                            sequence_id=sequence_id,
                            alias=alias.strip(),
                        ))

                except Exception as e:
                    print(f"❌ Skipped row due to error: {e}")

        print("✅ Data inserted successfully!")
