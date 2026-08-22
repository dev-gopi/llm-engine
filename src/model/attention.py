import torch
import torch.nn as nn
class MultiHeadAttention(nn.Module):
    def __init__(self,dim,heads=8):
        super().__init__()
        self.attn=nn.MultiheadAttention(dim,heads,batch_first=True)
    def forward(self,x):
        y,_=self.attn(x,x,x,need_weights=False)
        return y
