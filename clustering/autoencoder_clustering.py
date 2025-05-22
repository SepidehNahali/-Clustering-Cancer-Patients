import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon

def run_autoencoder_clustering(X_fused_feature, n_clusters=5, latent_dim=10):
    input_dim = X_fused_feature.shape[1]
    autoencoder = Autoencoder(input_dim, latent_dim)
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    tensor_x = torch.FloatTensor(X_fused_feature)
    loader = DataLoader(TensorDataset(tensor_x), batch_size=64, shuffle=True)

    for epoch in range(100):
        autoencoder.train()
        loss_sum = 0
        for batch in loader:
            batch_x = batch[0]
            optimizer.zero_grad()
            z, x_recon = autoencoder(batch_x)
            loss = criterion(x_recon, batch_x)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Autoencoder Epoch {epoch+1}, Loss: {loss_sum/len(loader):.4f}")

    autoencoder.eval()
    with torch.no_grad():
        z, _ = autoencoder(tensor_x)
    embeddings = z.numpy()

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    score = silhouette_score(embeddings, clusters)
    print("Autoencoder + K-Means Silhouette Score:", score)
    return clusters, score
