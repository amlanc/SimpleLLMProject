from torch import nn
from src.chapter03.MultiHeadAttention import MultiHeadAttention
from src.chapter04.LayerNorm import LayerNorm
from src.chapter04.SimpleFeedForward import SimpleFeedForward

# This is a basic representation of a Transformer block as used in an LLM


class TransformerBlock(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.sff = SimpleFeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # First use the norm->attention->dropout/shortcut
        # Left or bottom or Begin
        shortcut = x          # Store input values
        x = self.norm1(x)     # Normalize
        x = self.att(x)      # Multihead Attention Layer
        x = self.dropout(x)   # Dropout
        x += shortcut         # Shortcut

        # Then use the norm->FeedForward->dropout/shortcut
        shortcut = x
        x = self.norm2(x)    # Normalize
        x = self.sff(x)      # Feedfwd i.e. Linear->GELU->Linear
        x = self.dropout(x)  # Dropout
        x += shortcut        # Shortcut
        # Return the final value
        return x
