# 표준 라이브러리
import numpy as np
import pandas as pd
import networkx as nx
# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
# PyTorch Geometric
from torch_geometric.data import Data
from torch_geometric.nn import GATConv


# 1️⃣ 데이터 불러오기
df = pd.read_csv("fastapi\data\train_data_small.csv")

# 1. 노드 ID 추출 및 인덱싱
all_proteins = pd.unique(df[['protein1', 'protein2']].values.ravel())
protein_to_idx = {protein: idx for idx, protein in enumerate(all_proteins)}

# 2. edge_index 생성
edges = df[['protein1', 'protein2']].dropna().values
edge_index = torch.tensor([[protein_to_idx[a], protein_to_idx[b]] for a, b in edges], dtype=torch.long).T

# 3. 자동 피처 생성
df_feat = df.copy()

# 2️⃣ 결측값 처리
df_feat['sequence'] = df_feat['sequence'].fillna('')
df_feat['GO_Terms'] = df_feat['GO_Terms'].fillna('')
df_feat['PubMed_IDs'] = df_feat['PubMed_IDs'].fillna('')
df_feat['globalMetricValue'] = df_feat['globalMetricValue'].fillna(0)
df_feat['organismScientificName'] = df_feat.get('organismScientificName', pd.Series(['Unknown'] * len(df_feat))).fillna('Unknown')

# 수치형 피처 생성
df_feat['sequence_length'] = df_feat['sequence'].fillna('').apply(len)
df_feat['go_term_count'] = df_feat['GO_Terms'].fillna('').apply(lambda x: len(x.split(';')) if x else 0)
df_feat['pubmed_count'] = df_feat['PubMed_IDs'].fillna('').apply(lambda x: len(str(x).split(';')) if pd.notnull(x) else 0)

# 그래프 기반 피처
ppi_edges = df[['protein1', 'protein2', 'combined_score']].dropna()
ppi_graph = nx.from_pandas_edgelist(ppi_edges, source='protein1', target='protein2', edge_attr='combined_score')

ppi_degree = {prot: val for prot, val in ppi_graph.degree()}
ppi_avg_score = {node: np.mean([d['combined_score'] for _, _, d in ppi_graph.edges(node, data=True)]) for node in ppi_graph.nodes()}

df_feat['ppi_degree'] = df_feat['protein1'].map(ppi_degree).fillna(0)
df_feat['ppi_avg_score'] = df_feat['protein1'].map(ppi_avg_score).fillna(0)

df_feat['pdb_count'] = df_feat['PDB_IDs'].fillna('').apply(lambda x: len(x.split(';')) if x else 0)

gene_disease_count = df_feat.groupby('Gene ID')['Disease ID'].nunique()
df_feat['gene_disease_count'] = df_feat['Gene ID'].map(gene_disease_count).fillna(0)

# 수치형 스케일링
numerical_features = ['sequence_length', 'go_term_count', 'pubmed_count', 'globalMetricValue','ppi_degree', 'ppi_avg_score', 'pdb_count', 'gene_disease_count']
scaler = StandardScaler()
scaled_numerical = scaler.fit_transform(df_feat[numerical_features])

# 범주형 처리
categorical_features = ['organismScientificName']
df_feat[categorical_features] = df_feat[categorical_features].fillna('Unknown')
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_categorical = encoder.fit_transform(df_feat[categorical_features])

# 병합
feature_matrix = np.hstack([scaled_numerical, encoded_categorical])
df_feat['protein_index'] = df_feat['protein1'].map(protein_to_idx)

# 단백질별 평균 피처 생성
node_features = np.zeros((len(protein_to_idx), feature_matrix.shape[1]))
counts = df_feat['protein_index'].value_counts().to_dict()

for idx, row in df_feat.iterrows():
    p_idx = row['protein_index']
    if pd.notnull(p_idx):
        node_features[int(p_idx)] += feature_matrix[idx]

for prot, idx in protein_to_idx.items():
    count = counts.get(idx, 1)
    node_features[idx] /= count

x = torch.tensor(node_features, dtype=torch.float)

# 라벨 생성 (protein1에 포함되면 1, 아니면 0)
labeled_proteins = set(df['protein1'].dropna())
y = torch.tensor([1 if prot in labeled_proteins else 0 for prot in all_proteins], dtype=torch.long)

# PyG 데이터 객체
data = Data(x=x, edge_index=edge_index, y=y)
data

# GAT 모델 정의
class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        return x


from sklearn.model_selection import train_test_split

# 노드 인덱스 기준으로 train/test 나누기
idx = list(range(data.num_nodes))
train_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=data.y, random_state=42)

train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
train_mask[train_idx] = True
test_mask[test_idx] = True

data.train_mask = train_mask
data.test_mask = test_mask

# 하이퍼파라미터
in_channels = data.num_node_features
hidden_channels = 64
out_channels = len(torch.unique(data.y))

model = GAT(in_channels, hidden_channels, out_channels, heads=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# 학습 루프
for epoch in range(1, 101):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    # 평가
    model.eval()
    pred = out.argmax(dim=1)
    correct = pred[data.test_mask] == data.y[data.test_mask]
    acc = int(correct.sum()) / int(data.test_mask.sum())

    print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Test Acc: {acc:.4f}")