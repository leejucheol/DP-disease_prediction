import pandas as pd

# 원본 로드
df = pd.read_csv("./models/data/processed_train_small.csv")

# Proteins 테이블
df_protein = df[['UniProt_ID', 'sequence']].drop_duplicates().dropna()

# GO Term 테이블
df_go = df[['UniProt_ID', 'GO_Terms']].dropna()
df_go = df_go.assign(GO_Term=df_go['GO_Terms'].str.split(';')).explode('GO_Term')[['UniProt_ID', 'GO_Term']].drop_duplicates()

# Disease 테이블
df_disease = df[['UniProt_ID', 'Disease ID']].dropna()
df_disease = df_disease.assign(Disease_ID=df_disease['Disease ID'].str.split(';')).explode('Disease_ID')[['UniProt_ID', 'Disease_ID']].drop_duplicates()

# PDB 테이블
df_pdb = df[['UniProt_ID', 'PDB_IDs']].dropna()
df_pdb = df_pdb.assign(PDB_ID=df_pdb['PDB_IDs'].str.split(';')).explode('PDB_ID')[['UniProt_ID', 'PDB_ID']].drop_duplicates()

# Feature 테이블 (수치형 컬럼만 골라서)
exclude_cols = ['GO_Terms', 'sequence', 'PDB_IDs', 'Disease ID']
feature_cols = [col for col in df.columns if col not in exclude_cols]
df_features = df[feature_cols].drop_duplicates('UniProt_ID')

df_protein.to_csv("./models/data/protein.csv", index=False)
df_go.to_csv("./models/data/protein_go.csv", index=False)
df_disease.to_csv("./models/data/protein_disease.csv", index=False)
df_pdb.to_csv("./models/data/protein_pdb.csv", index=False)
df_features.to_csv("./models/data/protein_features.csv", index=False)
