import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        out = self.classifier(x)
        return out


class Main:
    @staticmethod
    def load_data(path='./data/train_data.csv'):
        df = pd.read_csv(path)
        print(f">>> 데이터 로드 성공, 데이터 크기: {df.shape}")
        return df

    @staticmethod
    def encode_proteins(df):
        proteins = pd.concat([df['protein1'], df['protein2']]).unique()
        encoder = LabelEncoder()
        encoder.fit(proteins)
        df['p1_idx'] = encoder.transform(df['protein1'])
        df['p2_idx'] = encoder.transform(df['protein2'])
        return df, encoder

    @staticmethod
    def build_edge_index(df):
        edge_index_np = np.array([df['p1_idx'].values, df['p2_idx'].values])
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)
        return edge_index

    @staticmethod
    def create_node_features(num_nodes, feature_dim=128):
        x = torch.rand((num_nodes, feature_dim))
        return x

    @staticmethod
    def run():
        df = Main.load_data()
        df, encoder = Main.encode_proteins(df)
        edge_index = Main.build_edge_index(df)
        x = Main.create_node_features(num_nodes=len(encoder.classes_))

        model = GCN(num_node_features=x.shape[1], hidden_channels=64, num_classes=10)
        out = model(x, edge_index)

        print(">>> 출력 크기:", out.shape)


if __name__ == "__main__":
    Main.run()
