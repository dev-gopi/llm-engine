"""Native encoder-decoder sequence-to-sequence Transformer architecture."""
from __future__ import annotations
import torch
from torch import Tensor, nn

class EncoderDecoderTransformer(nn.Module):
    """Compact native seq2seq model with separate source/target embeddings and LM head."""
    def __init__(self, vocab_size:int, d_model:int=256, nhead:int=8, layers:int=4, ff_dim:int=1024, max_position:int=2048, dropout:float=0.0):
        super().__init__()
        if min(vocab_size,d_model,nhead,layers,ff_dim,max_position)<1 or d_model%nhead: raise ValueError("invalid seq2seq dimensions")
        self.vocab_size=vocab_size; self.d_model=d_model; self.max_position=max_position
        self.src_embedding=nn.Embedding(vocab_size,d_model); self.tgt_embedding=nn.Embedding(vocab_size,d_model)
        self.src_pos=nn.Embedding(max_position,d_model); self.tgt_pos=nn.Embedding(max_position,d_model)
        layer_e=nn.TransformerEncoderLayer(d_model,nhead,ff_dim,dropout,batch_first=True,norm_first=True)
        layer_d=nn.TransformerDecoderLayer(d_model,nhead,ff_dim,dropout,batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer_e,layers); self.decoder=nn.TransformerDecoder(layer_d,layers); self.norm=nn.LayerNorm(d_model); self.lm_head=nn.Linear(d_model,vocab_size,bias=False)
    def forward(self, source_ids:Tensor, target_ids:Tensor, *, source_mask:Tensor|None=None, target_mask:Tensor|None=None):
        if source_ids.ndim!=2 or target_ids.ndim!=2: raise ValueError("source_ids and target_ids must be [batch,time]")
        if source_ids.shape[1]>self.max_position or target_ids.shape[1]>self.max_position: raise ValueError("sequence exceeds max_position")
        s=torch.arange(source_ids.shape[1],device=source_ids.device); t=torch.arange(target_ids.shape[1],device=target_ids.device)
        memory=self.encoder(self.src_embedding(source_ids)+self.src_pos(s)[None,:,:],src_key_padding_mask=(~source_mask.bool()) if source_mask is not None else None)
        causal=nn.Transformer.generate_square_subsequent_mask(target_ids.shape[1],device=target_ids.device)
        hidden=self.decoder(self.tgt_embedding(target_ids)+self.tgt_pos(t)[None,:,:],memory,tgt_mask=causal,tgt_key_padding_mask=(~target_mask.bool()) if target_mask is not None else None, memory_key_padding_mask=(~source_mask.bool()) if source_mask is not None else None)
        return self.lm_head(self.norm(hidden))
