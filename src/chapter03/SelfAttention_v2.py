import torch
import torch.nn as nn
from torch import Tensor


class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.device = _get_device()
        #
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        #
        # self.W_query.to(self.device)
        # self.W_key.to(self.device)
        # self.W_value.to(self.device)

    def forward(self, x):
        # x.to(self.device)
        print(f"x = {x.shape}")

        # self.W_query.to(self.device)
        # print("Moved W_query to mps")

        queries = self.W_query(x)
        # queries.to(self.device)
        # print("Moved queries to mps")
        print(f"queries.shape = {queries.shape}")

        keys = self.W_key(x)
        # keys.to(self.device)
        print(f"keys.shape = {keys.shape}")

        values = self.W_value(x)
        # values.to(self.device)
        print(f"values.shape = {values.shape}")

        attn_scores = queries @ keys.T  # Omega
        # attn_scores.to(self.device)
        print(f"attn_scores.shape = {attn_scores.shape}")

        attn_weights = torch.softmax(attn_scores / (keys.shape[-1] ** 0.5), dim=-1)
        # attn_weights.to(self.device)
        print(f"attn_weights.shape = {attn_weights.shape}")

        context_vec = attn_weights @ values
        # context_vec.to(self.device)
        print(f"context_vec shape = {context_vec.shape}")
        return context_vec


def _get_device():
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
        x: Tensor = torch.ones(1, device=device)
        print(f"x = {x} using 'cuda:0' backend")

    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        # print(mps_device)
        x: Tensor = torch.ones(1, device=device)
        print(f"x = {x} using 'mps:0' backend")
    else:
        device = torch.device("cpu")
        x: Tensor = torch.ones(1, device=device)
    return device
