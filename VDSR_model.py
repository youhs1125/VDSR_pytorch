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
    )
  def forward(self, X):
    residual = X.clone().detach()
    X = self.inputLayer(X)
    X = self.midLayer(X)
    X = self.outputLayer(X)
    X = torch.add(X,residual)
    return X


if __name__ == "__main__":
    model = VDSR()
    ten = torch.rand(size=(1,3,80,80))
    pred = model(ten)
    print(model)
    print(pred.shape)