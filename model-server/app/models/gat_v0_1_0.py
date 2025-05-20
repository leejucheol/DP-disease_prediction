import pandas as pd
import torch
import os
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from sklearn.preprocessing import LabelEncoder
from torch_geometric.nn import GATConv, global_mean_pool
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import BCELoss
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import esm

# 1. 데이터 로드
# 현재 파일의 경로를 기준으로 데이터 파일 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.dirname(os.path.dirname(current_dir))
data_path = os.path.join(app_dir, "app", "data", "processed_train_with_unchar.small.csv")

# 데이터 파일 존재 확인
if not os.path.exists(data_path):
    # 대체 경로 시도
    data_path = os.path.join(os.path.dirname(current_dir), "data", "processed_train_with_unchar.small.csv")

if os.path.exists(data_path):
    print(f"데이터 파일 로드: {data_path}")
else:
    print(f"데이터 파일을 찾을 수 없습니다: {data_path}")
    
df = pd.read_csv(data_path)

# 1. 모델 로드
esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
esm_model = esm_model.to(device)
esm_model.eval()

df.columns

# 서열을 그래프로 나타낸다 esm 임베딩해서
def sequence_to_graph_with_esm(seq, esm_model, batch_converter, device):
    seq = seq.strip().upper()
    batch = [("protein", seq)]
    _, _, tokens = batch_converter(batch)

    with torch.no_grad():
        tokens = tokens.to(device)
        results = esm_model(tokens, repr_layers=[6], return_contacts=False)
        token_embeddings = results["representations"][6][0, 1:-1]  # exclude CLS, EOS
        x = token_embeddings.cpu()

    edge_index = torch.tensor(
        [[i, i+1] for i in range(len(seq) - 1)] + [[i+1, i] for i in range(len(seq) - 1)],
        dtype=torch.long
    ).t()

    return Data(x=x, edge_index=edge_index)

# uniprotid와 질병 아이디 그룹화 질병 아이디를 y로 두고
from sklearn.preprocessing import MultiLabelBinarizer

df = df.drop('UniProt_ID', axis=1)

def generate_labels(df):
    grouped = df.groupby("sequence")["Disease ID"].apply(
        lambda x: list(set(x.dropna()))
    ).reset_index()
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(grouped["Disease ID"])
    # label_map의 value를 torch.tensor로 저장 (float, 1D)
    label_map = {uid: torch.tensor(y_row, dtype=torch.float) for uid, y_row in zip(grouped["sequence"], y)}
    return label_map, mlb

def build_graph_dataset_with_esm(df, esm_model, batch_converter, device):
    from torch_geometric.data import Data
    import numpy as np

    label_map, mlb = generate_labels(df)
    data_list = []

    uids, embeddings = [], []

    for row in df[["sequence"]].drop_duplicates().itertuples():
        seq = row.sequence
        uid = seq
        if not isinstance(seq, str) or len(seq) < 5:
            continue
        try:
            g = sequence_to_graph_with_esm(seq, esm_model, batch_converter, device)
            uids.append(uid)
            embeddings.append(g.x.cpu().numpy())
        except Exception as e:
            print(f"❌ ESM 실패: {uid} | {e}")

    for uid, emb in zip(uids, embeddings):
        try:
            x = torch.tensor(emb, dtype=torch.float)
            edge_index = torch.tensor(
                [[i, i + 1] for i in range(len(emb) - 1)] + [[i + 1, i] for i in range(len(emb) - 1)],
                dtype=torch.long
            ).t()

            g = Data(x=x, edge_index=edge_index)
            g.uid = uid

            if uid in label_map:
                g.y = torch.tensor(label_map[uid], dtype=torch.float).unsqueeze(0)  # (1, C)
            else:
                g.y = torch.zeros(1, len(mlb.classes_), dtype=torch.float)



            data_list.append(g)

        except Exception as e:
            print(f"❌ 그래프 실패: {uid} | {e}")

    return data_list, mlb

from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

uids = df["sequence"].unique()
train_ids, test_ids = train_test_split(uids, test_size=0.2, random_state=42)
train_df = df[df["sequence"].isin(train_ids)].copy()
test_df = df[df["sequence"].isin(test_ids)].copy()

drop_cols = ["Entry_Name", "proteinFullNames", "PDB_IDs", "Gene ID"]
train_df.drop(columns=[col for col in drop_cols if col in train_df.columns], inplace=True)
test_df.drop(columns=[col for col in drop_cols if col in test_df.columns], inplace=True)

train_data_list, mlb = build_graph_dataset_with_esm(train_df, esm_model, batch_converter, device)
test_data_list, _ = build_graph_dataset_with_esm(test_df, esm_model, batch_converter, device)

train_loader = DataLoader(train_data_list, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data_list, batch_size=16, shuffle=False)

batch = next(iter(DataLoader(train_data_list, batch_size=16)))
print(batch.y.shape)  # ✅ 반드시 (16, C) 나와야 정상

"""# 모델 학습 및 예측
## 모델 정의
"""

class ProteinGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.gat2 = GATConv(hidden_channels*heads, hidden_channels, heads=1)
        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x)


num_disease_classes = len(mlb.classes_)
model = ProteinGAT(in_channels=320, hidden_channels=64, out_channels=num_disease_classes)

print("질병 클래스 수:", num_disease_classes)

from torch_geometric.loader import DataLoader
import torch.nn as nn

num_disease_classes = len(mlb.classes_)
model = ProteinGAT(in_channels=320, hidden_channels=64, out_channels=num_disease_classes).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0

    for i, batch in enumerate(loader):
        try:
            batch = batch.to(device)
            optimizer.zero_grad()

            # 모델 예측
            out = model(batch.x, batch.edge_index, batch.batch)

            # 디버그용 shape 출력
            if epoch == 0 and i == 0:  # 첫 에폭 첫 배치만 확인
                print(f"🧪 Batch {i+1}")
                print("✅ out.shape:", out.shape)
                print("✅ y.shape:", batch.y.shape)

            # 손실 계산 및 역전파
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        except Exception as e:
            print(f"❌ Batch {i+1} 실패: {e}")

    return total_loss / len(loader)

from sklearn.metrics import f1_score

# ✅ 평가 함수
def evaluate(model, loader, device):
    model.eval()
    y_true_list, y_pred_list = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            probs = torch.sigmoid(out)

            # ⚠️ 반드시 2D 텐서로 append
            y_true_list.append(batch.y.view(probs.shape).detach().cpu())
            y_pred_list.append((probs > 0.5).detach().cpu())

    y_true = torch.cat(y_true_list, dim=0).numpy()
    y_pred = torch.cat(y_pred_list, dim=0).numpy()

    print(f"✅ y_true shape: {y_true.shape}")
    print(f"✅ y_pred shape: {y_pred.shape}")
    return y_true, y_pred

print(f"📦 학습 데이터 수: {len(train_data_list)}")
print(f"📦 배치 수: {len(train_loader)}")

# 하나 꺼내서 shape 확인
sample_batch = next(iter(train_loader))
print(f"✅ sample y shape: {sample_batch.y.shape}")
print(f"✅ sample x shape: {sample_batch.x.shape}")

num_epochs = 100
for epoch in range(num_epochs):
    loss = train(model, train_loader, optimizer, criterion, device, epoch)
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss:.4f}")

def predict_top5_diseases(sequence, model, esm_model, batch_converter, mlb, device="cpu"):
    # 1. 단백질 서열 → ESM 임베딩 → 그래프 변환
    graph = sequence_to_graph_with_esm(sequence, esm_model, batch_converter, device)
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long).to(device)

    # 2. 모델 추론
    model.eval()
    with torch.no_grad():
        out = model(graph.x, graph.edge_index, graph.batch)
        probs = torch.sigmoid(out).cpu().numpy().flatten()

    # 3. 상위 5개 인덱스 추출 (확률 내림차순)
    top5_idx = probs.argsort()[-5:][::-1]
    disease_id_to_name = dict(zip(df['Disease ID'], df['Disease Name']))
    top5_diseases = []
    for i in top5_idx:
        disease_id = mlb.classes_[i]
        probability = float(probs[i])
        disease_name = disease_id_to_name.get(disease_id, "Unknown")  # disease_id_to_name은 미리 정의된 dict여야 함
        top5_diseases.append({
            "disease_id": disease_id,
            "disease_name": disease_name,
            "probability": probability
        })
    return top5_diseases