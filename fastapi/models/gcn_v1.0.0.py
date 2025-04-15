import pandas as pd
import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 1️⃣ 데이터 불러오기
df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/DP/bio_datas/processed_train_small.csv")

# 매핑 테이블 생성
id_map = df[['UniProt_ID', 'protein1']].dropna().drop_duplicates()
protein1_to_uniprot = dict(zip(id_map['protein1'], id_map['UniProt_ID']))
# protein2 매핑 추가
id_map_2 = df[['UniProt_ID', 'protein2']].dropna().drop_duplicates()
protein2_to_uniprot = dict(zip(id_map_2['protein2'], id_map_2['UniProt_ID']))

# 두 개 합치기
combined_mapping = {**protein1_to_uniprot, **protein2_to_uniprot}

import torch
from torch_geometric.data import Data
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# 1. 노드 정의
proteins = pd.unique(df['UniProt_ID'])
protein_to_idx = {prot: idx for idx, prot in enumerate(proteins)}

# 2. 엣지 정의
edges = df[['protein1', 'protein2']].dropna()
edges['p1_mapped'] = edges['protein1'].map(combined_mapping)
edges['p2_mapped'] = edges['protein2'].map(combined_mapping)
edges_valid = edges.dropna(subset=['p1_mapped', 'p2_mapped'])

edge_index = torch.tensor([
    [protein_to_idx[p1], protein_to_idx[p2]]
    for p1, p2 in edges_valid[['p1_mapped', 'p2_mapped']].values
], dtype=torch.long).t()

# 3. 노드 특성 정의
df_nodes = df.drop_duplicates('UniProt_ID')[['UniProt_ID', 'GO_Terms', 'sequence', 'PDB_IDs']]
df_nodes = df_nodes.set_index('UniProt_ID').loc[proteins]

# 3-1. 서열 전처리 (공백 채움, 문자열로 변환)
df_nodes['sequence'] = df_nodes['sequence'].fillna('').astype(str) # fillna를 먼저하고 str로 바꿔야 빈 서열만 안전하게 문자열 ''로 처리할 수 있다.
# df_nodes['sequence'] = df_nodes['sequence'].astype(str).fillna('') 
# -> 타입을 str로 바꾸고 nan값이 nan으로 채워진다. 그리면 fillna는 아무 효과가 없다.결론적으로 nan이라는 이상한 서열이 들어가 잘못된 임베딩이라고 볼 수 있다.

# 3-2. GO_Terms → TF-IDF 임베딩
if df_nodes['GO_Terms'].str.strip().eq('').all():
    go_features = np.zeros((len(df_nodes), 10)) # Go_Term 자체가 매우 희소하다면 예외 처리를 추가하여 안정성을 높인다.

vectorizer = TfidfVectorizer()
go_features = vectorizer.fit_transform(df_nodes['GO_Terms'].fillna('')).toarray()

# 3-3. PDB 구조 여부 (0 or 1)
has_structure = (~df_nodes['PDB_IDs'].isna()).astype(int).values.reshape(-1, 1)

# 4. ESM 입력용 서열 데이터 생성
data_sequences = [
    (idx, seq) for idx, seq in df_nodes['sequence'].items() if pd.notnull(seq) and isinstance(seq, str)
]

# 단백질 서열을 임베딩하기 위해 라이브러리 설치
import esm

# ✅ 사전학습된 ESM 모델과 알파벳 불러오기
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()

# ✅ 배치 컨버터 생성 (서열 데이터를 토큰으로 바꿔주는 도구)
batch_converter = alphabet.get_batch_converter()

# 2. 모델을 GPU로 이동
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# 3. 단백질 서열 준비
# 문자열로 강제 변환 + 결측 필터링
data_sequences = [
    (idx, str(seq).strip())
    for idx, seq in df_nodes['sequence'].items()
    if isinstance(seq, str) or pd.notnull(seq)
]

# sequence 컬럼에 float/nan이 섞여 있는지 확인
print(df_nodes['sequence'].apply(type).value_counts())
print(df_nodes['sequence'].isna().sum())

batch_size = 16
embeddings_list = []

for i in range(0, len(data_sequences), batch_size):
    batch_seqs = data_sequences[i:i+batch_size]
    batch_labels, batch_strs, batch_tokens = batch_converter(batch_seqs)

    with torch.no_grad():
        batch_tokens = batch_tokens.to(device)
        results = model(batch_tokens, repr_layers=[6], return_contacts=False)
        emb = results["representations"][6][:, 0, :]  # CLS token
        embeddings_list.append(emb.cpu())

embeddings = torch.cat(embeddings_list, dim=0)
np.save("esm_embeddings.npy", embeddings.numpy())  # 저장

# 1. proteins 순서 기준
protein_ids = df_nodes.index.tolist()

# 2. ESM 결과를 dict로 인덱싱
esm_ids = [pid for pid, _ in data_sequences]
esm_index_map = {pid: i for i, pid in enumerate(esm_ids)}

# 3. 올바른 순서로 재정렬 (없는 건 0으로)
esm_features = np.zeros((len(protein_ids), embeddings.shape[1]))
for i, pid in enumerate(protein_ids):
    if pid in esm_index_map:
        esm_features[i] = embeddings[esm_index_map[pid]].numpy()

# 4. 안전하게 병합
x = np.hstack([go_features, has_structure, esm_features])
x_tensor = torch.tensor(x, dtype=torch.float)

# 각 파트별 feature shape 확인
print("✅ GO feature shape:", go_features.shape)        # 예: (N, 1000)
print("✅ 구조 여부 shape:", has_structure.shape)       # 예: (N, 1)
print("✅ ESM 임베딩 shape:", esm_features.shape)       # 예: (N, 320)

# 전체 병합한 x 벡터 shape 확인
print("📦 병합된 x 전체 shape:", x.shape)               # 예: (N, 1321)

# 검증: 차원 총합이 일치하는지
expected = go_features.shape[1] + has_structure.shape[1] + esm_features.shape[1]
print("🔍 총 피처 차원 일치 여부:", x.shape[1] == expected)

from sklearn.preprocessing import MultiLabelBinarizer
import torch
import numpy as np

# 1️⃣ 단백질별로 질병 리스트 구성
disease_per_protein = df.groupby("UniProt_ID")["Disease ID"].apply(
    lambda x: list(set(x.dropna()))
).reset_index()
disease_per_protein.columns = ["UniProt_ID", "Disease_IDs"]

# 2️⃣ MultiLabelBinarizer로 one-hot 라벨 생성
mlb = MultiLabelBinarizer()
y_multilabel = mlb.fit_transform(disease_per_protein["Disease_IDs"])

# 3️⃣ PyTorch 텐서 변환
y_multi = torch.tensor(y_multilabel, dtype=torch.float)

# 4️⃣ 질병 인덱스 → 질병 ID 매핑
idx_to_disease = {i: d for i, d in enumerate(mlb.classes_)}
disease_to_idx = {d: i for i, d in enumerate(mlb.classes_)}  # optional

# ✅ 확인
print("y shape:", y_multi.shape)
print("라벨이 있는 단백질 수:", y_multi.sum(dim=1).nonzero().size(0))
print("예시 idx_to_disease:", dict(list(idx_to_disease.items())[:5]))

from torch_geometric.data import Data

# x_tensor, edge_index, y_multi는 이미 앞에서 만들었다고 가정
data = Data(
    x=x_tensor,                  # 노드 특성 (GO + PDB + ESM 임베딩)
    edge_index=edge_index,      # 단백질 간 엣지
    y=y_multi                   # 단백질 → 질병 다중 라벨 (0/1)
)

# 질병 단백질 ID 목록 추출
# labeled_proteins = set(df['uniprotAccession'].dropna().unique())

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split

# ✅ GCN 모델 (multi-label)
class GCN_MultiLabel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)  # sigmoid로 각 질병별 확률 출력

# ✅ 라벨 생성 (multi-label): 단백질 → 질병 매핑 (0/1)
disease_ids = sorted(df['Disease ID'].dropna().unique().tolist())
disease_to_idx = {d: i for i, d in enumerate(disease_ids)}

y_multi = torch.zeros((len(proteins), len(disease_ids)))
for i, prot in enumerate(proteins):
    related_diseases = df[df['UniProt_ID'] == prot]['Disease ID'].dropna().unique()
    for d in related_diseases:
        if d in disease_to_idx:
            y_multi[i][disease_to_idx[d]] = 1

# 노드 인덱스 기준으로 train/test 나누기
idx = list(range(data.num_nodes))
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)

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

num_classes = len(torch.unique(data.y))
model = GCN_MultiLabel(
    in_channels=data.num_node_features,
    hidden_channels=64,
    out_channels=y_multi.shape[1]
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.BCELoss()

# 학습 루프
for epoch in range(1, 101):
    model.train()
    optimizer.zero_grad()

    out = model(data)  # shape: [num_nodes, num_diseases]
    loss = loss_fn(out[data.train_mask], y_multi[data.train_mask])
    loss.backward()
    optimizer.step()

    # 평가
    model.eval()
    with torch.no_grad():
        pred = (out[data.test_mask] > 0.5).int()
        true = y_multi[data.test_mask].int()
        acc = (pred == true).float().mean()
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Acc: {acc:.4f}")

def predict_node_multilabel(protein_id, model, data, protein_to_idx, idx_to_disease, threshold=0.5):
    model.eval()
    idx = protein_to_idx.get(protein_id)
    if idx is None:
        print("❌ Unknown protein ID:", protein_id)
        return [], []

    with torch.no_grad():
        output = model(data)[idx]
        probs = output.tolist()
        predicted = [idx_to_disease[i] for i, p in enumerate(probs) if p >= threshold]
        return predicted, probs

protein_id = 'Q9HAZ2'
predicted_diseases, probs = predict_node_multilabel(protein_id, model, data, protein_to_idx, idx_to_disease)

print(f"🔍 Protein: {protein_id}")
print(f"🎯 예측된 질병 ID들: {predicted_diseases}")
print(f"📊 확률 분포 (상위 5개): {sorted(probs, reverse=True)[:5]}")

predicted_disease_id = idx_to_disease[27]
print("예측 질병 ID:", predicted_disease_id)

edges = df[['protein1', 'protein2']].dropna()
valid_edges = [
    (protein_to_idx[p1], protein_to_idx[p2])
    for p1, p2 in edges.values if p1 in protein_to_idx and p2 in protein_to_idx
]

if len(valid_edges) == 0:
    print("엣지가 비어 있습니다! protein1, protein2 값 확인이 필요합니다.")
else:
    edge_index = torch.tensor(valid_edges, dtype=torch.long).t()
    print(f"edge_index shape: {edge_index.shape}")

proteins = set(proteins)
edge_proteins = set(edges['protein1']).union(set(edges['protein2']))

missing_proteins = edge_proteins - proteins
print(f"매칭되지 않은 protein 개수: {len(missing_proteins)}")
print("매칭되지 않은 protein 예시:", list(missing_proteins)[:10])