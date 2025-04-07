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


class ProteinDiseasePredictor:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.data = None

    def preprocess_data(self):
        df = self.df.copy()
        all_proteins = pd.unique(df[['protein1', 'protein2']].values.ravel())
        protein_to_idx = {protein: idx for idx, protein in enumerate(all_proteins)}
        edges = df[['protein1', 'protein2']].dropna().values
        edge_index = torch.tensor([[protein_to_idx[a], protein_to_idx[b]] for a, b in edges], dtype=torch.long).T

        # 결측값 처리 및 피처 생성
        df['sequence'] = df['sequence'].fillna('')
        df['GO_Terms'] = df['GO_Terms'].fillna('')
        df['PubMed_IDs'] = df['PubMed_IDs'].fillna('')
        df['globalMetricValue'] = df['globalMetricValue'].fillna(0)
        df['organismScientificName'] = df.get('organismScientificName', pd.Series(['Unknown'] * len(df))).fillna('Unknown')
        df['sequence_length'] = df['sequence'].apply(len)
        df['go_term_count'] = df['GO_Terms'].apply(lambda x: len(x.split(';')) if x else 0)
        df['pubmed_count'] = df['PubMed_IDs'].apply(lambda x: len(str(x).split(';')) if pd.notnull(x) else 0)

        # 그래프 기반 피처
        ppi_edges = df[['protein1', 'protein2', 'combined_score']].dropna()
        ppi_graph = nx.from_pandas_edgelist(ppi_edges, source='protein1', target='protein2', edge_attr='combined_score')
        ppi_degree = dict(ppi_graph.degree())
        ppi_avg_score = {node: np.mean([d['combined_score'] for _, _, d in ppi_graph.edges(node, data=True)]) for node in ppi_graph.nodes()}
        df['ppi_degree'] = df['protein1'].map(ppi_degree).fillna(0)
        df['ppi_avg_score'] = df['protein1'].map(ppi_avg_score).fillna(0)
        df['pdb_count'] = df['PDB_IDs'].fillna('').apply(lambda x: len(x.split(';')) if x else 0)
        gene_disease_count = df.groupby('Gene ID')['Disease ID'].nunique()
        df['gene_disease_count'] = df['Gene ID'].map(gene_disease_count).fillna(0)

        # 피처 정리
        numerical_features = ['sequence_length', 'go_term_count', 'pubmed_count', 'globalMetricValue',
                              'ppi_degree', 'ppi_avg_score', 'pdb_count', 'gene_disease_count']
        scaler = StandardScaler()
        scaled_numerical = scaler.fit_transform(df[numerical_features])

        categorical_features = ['organismScientificName']
        df[categorical_features] = df[categorical_features].fillna('Unknown')
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_categorical = encoder.fit_transform(df[categorical_features])

        feature_matrix = np.hstack([scaled_numerical, encoded_categorical])
        df['protein_index'] = df['protein1'].map(protein_to_idx)

        node_features = np.zeros((len(protein_to_idx), feature_matrix.shape[1]))
        counts = df['protein_index'].value_counts().to_dict()
        for idx, row in df.iterrows():
            p_idx = row['protein_index']
            if pd.notnull(p_idx):
                node_features[int(p_idx)] += feature_matrix[idx]
        for prot, idx in protein_to_idx.items():
            count = counts.get(idx, 1)
            node_features[idx] /= count

        x = torch.tensor(node_features, dtype=torch.float)
        labeled_proteins = set(df['protein1'].dropna())
        y = torch.tensor([1 if prot in labeled_proteins else 0 for prot in all_proteins], dtype=torch.long)
        self.data = Data(x=x, edge_index=edge_index, y=y)

    def train(self, heads=1, hidden_channels=64, epochs=100, lr=0.01):
        data = self.data
        idx = list(range(data.num_nodes))
        train_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=data.y, random_state=42)

        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        test_mask[test_idx] = True

        data.train_mask = train_mask
        data.test_mask = test_mask

        in_channels = data.num_node_features
        out_channels = len(torch.unique(data.y))
        model = GAT(in_channels, hidden_channels, out_channels, heads=heads)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            out = model(data)
            loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            model.eval()
            pred = out.argmax(dim=1)
            correct = pred[data.test_mask] == data.y[data.test_mask]
            acc = int(correct.sum()) / int(data.test_mask.sum())

            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Test Acc: {acc:.4f}")


# 실행 예시
# predictor = ProteinDiseasePredictor("train_data_small.csv")
# predictor.preprocess_data()
# predictor.train(heads=1)

