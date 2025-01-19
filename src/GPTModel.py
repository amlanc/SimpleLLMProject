import os

import torch
from torch import nn

from src.LayerNorm import LayerNorm
from src.TransformerBlock import TransformerBlock


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        if torch.backends.mps.is_available() and os.name == "posix":
            device = torch.device('mps:0')
        else:
            device = torch.device('cpu')

        self.device = device

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])

        # Exact same dim as it came in
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        # if torch.backends.mps.is_available():
        #     device = torch.device('mps')
        # print("Input Shape: ", in_idx.shape)

        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        # pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        print(f"device = {self.device}")
        pos_embeds = self.pos_emb(torch.arange(seq_len))

        # x = torch.cat((tok_embeds, pos_embeds), dim=-1)
        x = tok_embeds + pos_embeds

        x = self.drop_emb(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


