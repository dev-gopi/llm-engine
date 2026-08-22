import torch
import torch.nn as nn
class PositionalEmbedding(nn.Module):
    def __init__(self,max_pos,dim):
        super().__init__()
        self.emb=nn.Embedding(max_pos,dim)
    def forward(self,x):
        pos=torch.arange(x.size(1),device=x.device)
        return self.emb(pos)
