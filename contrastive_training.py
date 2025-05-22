import torch
import torch.nn as nn
from contrastive_loss import contrastive_loss

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.net(x)

def train_contrastive_model(view1, view2, epochs=100):
    model = MLPEncoder(view1.shape[1], 128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    view1 = torch.tensor(view1, dtype=torch.float32)
    view2 = torch.tensor(view2, dtype=torch.float32)

    for epoch in range(epochs):
        model.train()
        z1 = model(view1)
        z2 = model(view2)
        loss = contrastive_loss(z1, z2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

    return model
