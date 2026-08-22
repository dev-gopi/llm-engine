import torch.nn as nn
from .attention import MultiHeadAttention
from .feed_forward import FeedForward
class TransformerBlock(nn.Module):
    def __init__(self,dim,heads=8):
        super().__init__()
        self.attn=MultiHeadAttention(dim,heads)
        self.ffn=FeedForward(dim)
        self.n1=nn.LayerNorm(dim)
        self.n2=nn.LayerNorm(dim)
    def forward(self,x):
        x=self.n1(x+self.attn(x))
        x=self.n2(x+self.ffn(x))
        return x
