from torch import nn
from chapter04.GELU import GELU


class SimpleFeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Expand the dimentions from 768 to 3072 and contract back to 768
        # after applying GELU
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )

    def forward(self, x):
        return self.layers(x)
