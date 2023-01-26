import torch
import torch.nn as nn

class VDSR(nn.Module):
  def __init__(self):
    super().__init__()

    self.inputLayer = nn.Sequential(
        nn.Conv2d(3,64,kernel_size=3,padding=1),
        nn.ReLU()
    )
    mid = [nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.ReLU(),
        ) for _ in range(18)]
    self.midLayer = nn.Sequential(*mid)
    self.outputLayer = nn.Sequential(
        nn.Conv2d(64,3,kernel_size=3,padding=1),
        nn.ReLU()
    )
  def forward(self, X):
    residual = X.clone()

    X = self.inputLayer(X)
    X = self.midLayer(X)
    X = self.outputLayer(X)
    X = torch.add(X,residual)
    return X