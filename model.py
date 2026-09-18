# Deep SVDD Implementation
import torch
import torch.nn as nn

class DeepSVDDNetwork(nn.Module):
    def __init__(self, input_dim=64, rep_dim=32):
        super().__init__()
        self.rep_dim = rep_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, rep_dim)
        )
        self.c = None  # hypersphere center

    def forward(self, x):
        return self.net(x)

def init_center(model, loader, device="cpu", eps=0.1):
    model.eval()
    with torch.no_grad():
        outputs = [model(x.to(device)) for x, in loader]
        c = torch.cat(outputs).mean(dim=0)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        model.c = c

def svdd_loss(outputs, c):
    return torch.mean(torch.sum((outputs - c) ** 2, dim=1))
