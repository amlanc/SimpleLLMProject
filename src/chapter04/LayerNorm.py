import torch
from torch import nn


class LayerNorm(nn.Module):

    def __init__(self, emb_dim):
        super().__init__()
        # eps is a small constant (epsilon) added to the variance to
        # prevent division by zero during normalization.
        self.eps = 1e-5
        # The "scale" and "shift" are two trainable parameters
        # (of same dimension as the input) that the LLM adjusts during training
        # to improve the model’s performance
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        scaled_shifted = (norm_x * self.scale) + self.shift
        return scaled_shifted
