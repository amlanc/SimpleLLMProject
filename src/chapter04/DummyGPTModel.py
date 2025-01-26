import torch
import torch.nn as nn
import os


class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if torch.backends.mps.is_available() and os.name == "posix":
            device = torch.device('mps:0')
        else:
            device = torch.device('cpu')
        self.device = device
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"]).to(self.device)
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"]).to(self.device)
        self.drop_emb = nn.Dropout(cfg["drop_rate"]).to(self.device)

        # Unpack the DummyTransformerBlock objects initialized with config in the config array of n_layers
        # and create sequential layers of them
        self.trf_blocks = nn.Sequential(*[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = DummyLayerNorm(cfg["emb_dim"]).to(self.device)
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        ).to(self.device)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx).to(self.device)
        pos_embeds = self.pos_emb(torch.arange(seq_len).to(self.device)).to(self.device)
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        x.to(self.device)
        logits = self.out_head(x)
        return logits


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x


class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x
