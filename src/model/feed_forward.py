import torch.nn as nn
class FeedForward(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(dim,dim*4),
            nn.GELU(),
            nn.Linear(dim*4,dim)
        )
    def forward(self,x):
        return self.net(x)
