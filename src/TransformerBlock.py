import torch
from torch import nn
from MultiHeadAttention import MultiHeadAttention
from SimpleFeedForward import SimpleFeedForward

# This is a basic representation of a Transformer block as used in an LLM


class TransformerBlock(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.sff = SimpleFeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # First use the norm->attention->dropout/shortcut
        # Left or bottom or Begin
        shortcut = x          # Store input values
        x = self.norm1(x)     # Normalize
        x = self.attn(x)      # Multihead Attention Layer
        x = self.dropout(x)   # Dropout
        x += shortcut         # Shortcut

        # Then use the norm->FeedForward->dropout/shortcut
        # Right or Top or End
        shortcut = x
        x = self.norm2(x)    # Normalize
        x = self.sff(x)      # Feedfwd i.e. Linear->GELU->Linear
        x = self.dropout(x)  # Dropout
        x += shortcut        # Shortcut
        # Return the final value
        return x
