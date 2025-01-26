import os
import torch
from torch import nn
from src.chapter04.LayerNorm import LayerNorm
from src.chapter04.TransformerBlock import TransformerBlock


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if torch.backends.mps.is_available() and os.name == "posix":
            device = torch.device('mps:0')
        else:
            device = torch.device('cpu')
        self.device = device
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"]).to(device=self.device)
        # print(f"__init__(): tok_emb created")
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"]).to(device=self.device)
        # print(f"__init__(): pos_emb created")
        self.drop_emb = nn.Dropout(cfg["drop_rate"]).to(device=self.device)
        # print(f"__init__(): dropout layer created")
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        ).to(device=self.device)
        # print(f"__init__(): transformers created")
        self.final_norm = LayerNorm(cfg["emb_dim"])
        # print(f"__init__(): LayerNorm created")
        # Exact same dim as it came in
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
        # print(f"__init__(): Linear out projection layer created\n")

    def forward(self, in_idx):
        # if torch.backends.mps.is_available():
        #     self.device = torch.device('mps')
        batch_size, seq_len = in_idx.shape
        print(f"forward(): batch_size: {batch_size}, seq_len: {seq_len} and Input Shape: {in_idx.shape}")

        in_idx.to(self.device)
        tok_embeds = self.tok_emb(in_idx).to(self.device)
        print("forward(): created tok_embeds")

        pos_embeds = self.pos_emb(
            torch.arange(seq_len).to(self.device)
        ).to(self.device)
        print("forward(): created pos_embeds")

        # x = torch.cat((tok_embeds, pos_embeds), dim=-1)
        x = tok_embeds + pos_embeds

        print(f"forward(): x.shape: {x.shape}")
        x = self.drop_emb(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        print(f"forward(): logits.shape: {logits.shape}")
        return logits
