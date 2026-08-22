import torch
import torch.nn as nn
from .embedding import TokenEmbedding
from .positional import PositionalEmbedding
from .transformer_block import TransformerBlock
class MiniGPT(nn.Module):
    def __init__(self,vocab_size,dim=128,layers=4,heads=4,max_pos=512):
        super().__init__()
        self.tok=TokenEmbedding(vocab_size,dim)
        self.pos=PositionalEmbedding(max_pos,dim)
        self.blocks=nn.Sequential(*[TransformerBlock(dim,heads) for _ in range(layers)])
        self.norm=nn.LayerNorm(dim)
        self.head=nn.Linear(dim,vocab_size)
    def forward(self,ids):
        x=self.tok(ids)+self.pos(ids)
        x=self.blocks(x)
        return self.head(self.norm(x))
