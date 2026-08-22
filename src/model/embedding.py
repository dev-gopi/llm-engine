import torch.nn as nn
class TokenEmbedding(nn.Module):
    def __init__(self,vocab_size,dim):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,dim)
    def forward(self,x):
        return self.embedding(x)
