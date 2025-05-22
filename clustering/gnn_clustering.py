import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, BatchNorm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        return self.fc(x)

def run_gnn_clustering(X_fused_feature, n_clusters=5):
    threshold = np.percentile(X_fused_feature, 90)
    edge_idx = np.array(np.where(X_fused_feature > threshold))
    edge_index = torch.LongTensor(edge_idx)
    node_features = torch.FloatTensor(X_fused_feature)
    data = Data(x=node_features, edge_index=edge_index)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(in_channels=X_fused_feature.shape[1], hidden_channels=64, out_channels=24).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    data = data.to(device)

    for epoch in range(400):
        model.train()
        optimizer.zero_grad()
        z = model(data)
        loss = F.mse_loss(z, data.x)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"GNN Epoch {epoch+1}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        embeddings = model(data).cpu().numpy()

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    score = silhouette_score(embeddings, clusters)
    print("GNN Clustering Silhouette Score:", score)
    return clusters, score
