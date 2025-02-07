import os
import torch
from torch import nn
from chapter04.LayerNorm import LayerNorm
from chapter04.TransformerBlock import TransformerBlock


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if torch.backends.mps.is_available() and os.name == "posix":
            device = torch.device('mps:0')
        else:
            device = torch.device('cpu')
        self.device = device
        self.tok_emb = nn.Embedding(
            cfg["vocab_size"], cfg["emb_dim"]).to(self.device)
        self.pos_emb = nn.Embedding(
            cfg["context_length"], cfg["emb_dim"]).to(self.device)
        self.drop_emb = nn.Dropout(
            cfg["drop_rate"]).to(self.device)
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]).to(self.device)
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        in_idx.to(self.device)
        tok_embeds = self.tok_emb(in_idx).to(self.device)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len).to(self.device)).to(self.device)
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
