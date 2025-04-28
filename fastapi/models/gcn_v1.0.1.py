import pandas as pd
import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data
import torch
from torch_geometric.data import Data
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 1. 데이터 불러오기
print("========== 1. 데이터 로드 및 확인 ==========")

def load_data(path: str):
    try:
        df = pd.read_csv(path)
        if df.empty:
            print(">>> 데이터 비어있음.")
        else:
            print("✅ 데이터 로드 성공.")
            print(f">>> 데이터 크기: {df.shape}")
            print(f">>> 데이터 컬럼: {df.columns}\n")
            return df
    except:
        print(">>> ERROR: 데이터를 불러올 수 없음.")

df = load_data("./data/processed_train.csv")

# 각 단백질 ID (protein1)와 UniProt_ID 매핑 테이블
protein_to_uniprot = df[['protein1', 'UniProt_ID']].dropna().drop_duplicates()
protein_map = dict(zip(protein_to_uniprot['protein1'], protein_to_uniprot['UniProt_ID']))

all_proteins = pd.unique(df[['protein1', 'protein2']].values.ravel())
print(f">>> 전체 단백질 ID 수: {len(all_proteins)}")
print(f">>> protein1 ID 수: {len(df['protein1'].unique())}")
print(f">>> protein2 ID 수: {len(df['protein2'].unique())}")
print(f">>> 전체 Uniplot ID 수: {len(df['UniProt_ID'].unique())}")

print("\n========== 2. 매핑 시작 ==========")
# 1. protein1 → UniProt_ID 매핑 테이블 만들기
protein_to_uniprot = df[['protein1', 'UniProt_ID']].dropna().drop_duplicates()
protein_map = dict(zip(protein_to_uniprot['protein1'], protein_to_uniprot['UniProt_ID']))

# 2. protein1, protein2 각각에 매핑 적용
df['mapped_protein1'] = df['protein1'].map(protein_map)
df['mapped_protein2'] = df['protein2'].map(protein_map)

# 3. 유효한 edge만 필터링
valid_edges = df.dropna(subset=['mapped_protein1', 'mapped_protein2'])

print(f"✅ 매핑 성공")
print(f">>> 유효한 엣지 수: {len(valid_edges)}")

print("\n========== 3. 노드 인덱싱 및 edge_index 생성 ==========")

# 유효한 단백질 (노드): mapped_protein1, mapped_protein2에서 추출
all_nodes = pd.unique(valid_edges[['mapped_protein1', 'mapped_protein2']].values.ravel())
node_to_idx = {prot: idx for idx, prot in enumerate(sorted(all_nodes))}
print(f">>> 노드 개수 (고유 UniProt_ID): {len(all_nodes)}")

# edge_index 구성
edge_index = torch.tensor([
    [node_to_idx[row['mapped_protein1']], node_to_idx[row['mapped_protein2']]]
    for _, row in valid_edges.iterrows()
], dtype=torch.long).t()

print(f">>> edge_index shape: {edge_index.shape}")
print(f">>> edge_index 예시 (앞 5개):\n{edge_index[:, :5]}")

print("\n========== 4. 노드 특성 (x) 생성 ==========")

# 노드 정보: UniProt_ID 기준으로 중복 제거
df_nodes = df.dropna(subset=['UniProt_ID']).drop_duplicates('UniProt_ID').set_index('UniProt_ID')

# 노드 순서 맞추기
df_nodes = df_nodes.loc[sorted(node_to_idx.keys())]

# 1. GO_Terms → TF-IDF
vectorizer = TfidfVectorizer()
go_features = vectorizer.fit_transform(df_nodes['GO_Terms'].fillna('')).toarray()

# 2. PDB_ID 존재 여부
pdb_flag = (~df_nodes['PDB_IDs'].isna()).astype(int).values.reshape(-1, 1)

# 3. feature 합치기
features = np.hstack([go_features, pdb_flag])
x = torch.tensor(features, dtype=torch.float)

print(f">>> 노드 특성 텐서 shape: {x.shape}")

print("\n========== 5. 질병 라벨 벡터 생성 ==========")

from sklearn.preprocessing import MultiLabelBinarizer

# UniProt_ID 별로 연결된 Disease ID 집합 만들기
protein_disease_map = df.groupby('UniProt_ID')['Disease ID'].apply(set)

# 노드 순서에 맞춰 재정렬
protein_disease_map = protein_disease_map.reindex(sorted(node_to_idx.keys())).fillna(set())

# multi-label 인코딩
mlb = MultiLabelBinarizer()
y = torch.tensor(mlb.fit_transform(protein_disease_map), dtype=torch.float)

print(f">>> 라벨 벡터 shape: {y.shape}")
print(f">>> 질병 클래스 수: {len(mlb.classes_)}")


print("\n========== 6. PyTorch Geometric Data 객체 생성 ==========")

data = Data(
    x=x,
    edge_index=edge_index,
    y=y
)

print(data)
