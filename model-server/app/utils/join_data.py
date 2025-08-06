import pandas as pd

# 정규화된 테이블들 로드 또는 재사용
df_protein = pd.read_csv("./models/data/protein.csv")
df_go = pd.read_csv("./models/data/protein_go.csv")
df_disease = pd.read_csv("./models/data/protein_disease.csv")
df_pdb = pd.read_csv("./models/data/protein_pdb.csv")
df_features = pd.read_csv("./models/data/protein_features.csv")

# GO_Terms 재집계 (UniProt_ID별로 세미콜론으로 병합)
df_go_agg = df_go.groupby("UniProt_ID")["GO_Term"].apply(lambda x: ";".join(sorted(set(x)))).reset_index()
df_go_agg.rename(columns={"GO_Term": "GO_Terms"}, inplace=True)

# Disease ID 재집계
df_disease_agg = df_disease.groupby("UniProt_ID")["Disease_ID"].apply(lambda x: ";".join(sorted(set(x)))).reset_index()
df_disease_agg.rename(columns={"Disease_ID": "Disease ID"}, inplace=True)

# PDB ID 재집계
df_pdb_agg = df_pdb.groupby("UniProt_ID")["PDB_ID"].apply(lambda x: ";".join(sorted(set(x)))).reset_index()
df_pdb_agg.rename(columns={"PDB_ID": "PDB_IDs"}, inplace=True)

# 병합 수행
df_joined = df_protein \
    .merge(df_go_agg, on="UniProt_ID", how="left") \
    .merge(df_disease_agg, on="UniProt_ID", how="left") \
    .merge(df_pdb_agg, on="UniProt_ID", how="left") \
    .merge(df_features, on="UniProt_ID", how="left")

# 결과 저장
df_joined.to_csv("./models/data/reconstructed_processed_train.csv", index=False)
