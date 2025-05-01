import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from sklearn.preprocessing import LabelEncoder
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import BCELoss
from sklearn.metrics import roc_auc_score, average_precision_score

# 1️⃣ CSV 읽기
df = pd.read_csv("./data/processed_train.csv")
print(f"✅ 데이터 로드 성공: {df.shape}")

# 2️⃣ 노드 목록 추출
proteins = sorted(set(df["protein1"]) | set(df["protein2"]) | set(df["UniProt_ID"]))
diseases = sorted(df["Disease ID"].dropna().unique())
print(f"✅ 단백질 노드 개수: {len(proteins)}, 질병 노드 개수: {len(diseases)}")

# 3️⃣ 노드 → 정수 인덱스 매핑
protein_encoder = LabelEncoder().fit(proteins)
disease_encoder = LabelEncoder().fit(diseases)
num_proteins, num_diseases = len(proteins), len(diseases)
num_nodes = num_proteins + num_diseases
print(f"✅ 전체 노드 수: {num_nodes}")

def to_node_idx(id_, is_protein=True):
    if is_protein:
        return int(protein_encoder.transform([id_])[0])
    else:
        return num_proteins + int(disease_encoder.transform([id_])[0])

# 4️⃣ 엣지 리스트 생성
ppi_edges = df[["protein1","protein2"]].dropna().drop_duplicates().values
ppi_index_list = (
    [[to_node_idx(u,True), to_node_idx(v,True)] for u,v in ppi_edges] +
    [[to_node_idx(v,True), to_node_idx(u,True)] for u,v in ppi_edges]
)
ppi_edge_index = torch.tensor(ppi_index_list, dtype=torch.long).t().contiguous()
print(f"✅ PPI edge_index shape: {ppi_edge_index.shape}")

assoc = df[["UniProt_ID","Disease ID"]].dropna().drop_duplicates().values
assoc_index_list = (
    [[to_node_idx(p,True), to_node_idx(d,False)] for p,d in assoc] +
    [[to_node_idx(d,False), to_node_idx(p,True)] for p,d in assoc]
)
assoc_edge_index = torch.tensor(assoc_index_list, dtype=torch.long).t().contiguous()
print(f"✅ Association edge_index shape: {assoc_edge_index.shape}")

# 5️⃣ 노드 특성 준비 (더미)
x = torch.ones((num_nodes, 1), dtype=torch.float)
print(f"✅ Node feature matrix x shape: {x.shape}")

# 6️⃣ Data 객체 생성
edge_index = torch.cat([ppi_edge_index, assoc_edge_index], dim=1)
data = Data(x=x, edge_index=edge_index)
print(f"✅ Data 객체: num_nodes={data.num_nodes}, num_edges={data.num_edges}")

# 7️⃣ GCN 모델 정의
class ProteinDiseaseGCN(torch.nn.Module):
    def __init__(self, in_feats, hidden, out_feats):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden)
        self.conv2 = GCNConv(hidden, out_feats)
    def forward(self, data):
        h = F.relu(self.conv1(data.x, data.edge_index))
        return self.conv2(h, data.edge_index)

model = ProteinDiseaseGCN(in_feats=1, hidden=64, out_feats=32)
with torch.no_grad():
    z = model(data)
print(f"✅ 초기 forward 완료: node embedding shape {z.shape}")

# 8️⃣ 링크 예측용 Data split (split_labels=True 중요)
transform = RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    is_undirected=True,
    add_negative_train_samples=True,
    split_labels=True,        # ← pos/neg를 분리
    key='edge_label',
)
train_data, val_data, test_data = transform(data)

print(f"✅ Train pos edges: {train_data.pos_edge_label_index.size(1)}, "
        f"neg edges: {train_data.neg_edge_label_index.size(1)}")
print(f"   Val   pos edges: {val_data.pos_edge_label_index.size(1)}, "
        f"neg edges: {val_data.neg_edge_label_index.size(1)}")
print(f"   Test  pos edges: {test_data.pos_edge_label_index.size(1)}, "
        f"neg edges: {test_data.neg_edge_label_index.size(1)}")

# 9️⃣ LinkPredictor 정의
class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_channels * 2, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)
    def forward(self, z, edge_index):
        src, dst = edge_index
        h = torch.cat([z[src], z[dst]], dim=1)
        h = F.relu(self.lin1(h))
        return torch.sigmoid(self.lin2(h)).view(-1)

pred_head = LinkPredictor(in_channels=32, hidden_channels=16)

# 10️⃣ 학습 준비
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, pred_head = model.to(device), pred_head.to(device)
optimizer = Adam(list(model.parameters()) + list(pred_head.parameters()), lr=1e-3)
criterion = BCELoss()

# 11️⃣ 학습 함수 (pos/neg 속성명 수정)
def train():
    model.train(); pred_head.train()
    optimizer.zero_grad()
    z = model(train_data.to(device))
    pos_out = pred_head(z, train_data.pos_edge_label_index.to(device))
    neg_out = pred_head(z, train_data.neg_edge_label_index.to(device))
    pos_label = torch.ones(pos_out.size(0), device=device)
    neg_label = torch.zeros(neg_out.size(0), device=device)
    out = torch.cat([pos_out, neg_out], dim=0)
    label = torch.cat([pos_label, neg_label], dim=0)
    loss = criterion(out, label)
    loss.backward()
    optimizer.step()
    return loss.item()

# 12️⃣ 평가 함수 (val/test도 동일하게)
@torch.no_grad()
def evaluate(split_data):
    model.eval(); pred_head.eval()
    z = model(split_data.to(device))
    pos = pred_head(z, split_data.pos_edge_label_index.to(device))
    neg = pred_head(z, split_data.neg_edge_label_index.to(device))
    y_true = torch.cat([torch.ones(pos.size(0)), torch.zeros(neg.size(0))]).cpu().numpy()
    y_score = torch.cat([pos, neg], dim=0).cpu().numpy()
    return {
        'auc': roc_auc_score(y_true, y_score),
        'ap':  average_precision_score(y_true, y_score)
    }

# 13️⃣ 학습 루프 실행
for epoch in range(1, 51):
    loss = train()
    print(f"Epoch {epoch:02d} — Loss: {loss:.4f}")
    if epoch % 5 == 0:
        metrics = evaluate(val_data)
        print(f"  ↳ Val AUC: {metrics['auc']:.4f}, Val AP: {metrics['ap']:.4f}")

print("✅ 학습 완료")
