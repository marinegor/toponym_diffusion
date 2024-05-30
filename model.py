import math

import torch
import torch.nn.functional as F
from torch import nn
from torchtyping import TensorType

T = TensorType


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)


def positional_embeding_matrix(n_tokens: int, d_out: int) -> T["n_tokens", "out_dim"]:  # noqa: F821
    assert d_out % 2 == 0, f"d_out should be even, got {d_out=}"
    position: T["n_tokens", 1] = torch.arange(n_tokens).unsqueeze(-1)
    div_term: T["d_out // 2"] = (
        torch.arange(0, d_out, 2) * (-math.log(10_000.0) / d_out)
    ).exp()
    pe = torch.zeros(n_tokens, d_out)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class TimeEmbedding(nn.Module):
    def __init__(
        self,
        max_T: int,
        d_embed: int,
        d_output: int,
    ):
        super().__init__()
        self.max_T = max_T
        self.d_embed = d_embed
        self.d_output = d_output

        pe: T["max_T", "d_embed"] = positional_embeding_matrix(max_T, d_output)
        self.register_buffer("pe", pe)

    def forward(
        self,
        x: T["b", "max_T", torch.long],  # noqa: F821
    ) -> T["b", "max_T", "d_output"]:  # noqa: F821
        assert x.dtype == torch.long, f"{x.dtype=}"
        pos: T["b", "self.d_embed"] = self.pe[x]  # noqa: F821
        return pos


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        n_tokens: int,
        d_output: int,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_output = d_output

        self.embedding = nn.Embedding(
            num_embeddings=self.n_tokens, embedding_dim=d_output
        )

        pe: T["n_tokens", "d_output"] = positional_embeding_matrix(n_tokens, d_output)
        self.register_buffer("pe", pe)

    def forward(
        self,
        x: T["b", "n_tokens", torch.long],  # noqa: F821
    ) -> T["b", "n_tokens", "d_output"]:  # noqa: F821
        assert x.dtype == torch.long, f"{x.dtype=}"
        pos: T["b", "self.d_embed"] = self.pe[x]  # noqa: F821
        emb: T["b", "self.d_embed"] = self.embedding(x)  # noqa: F821
        return emb + pos


class SelfAttention(nn.Module):
    def __init__(self, d_in: int, d_out_kq: int, d_out_v: int):
        super().__init__()

        self.d_in = d_in
        self.d_out_kq = d_out_kq
        self.d_out_v = d_out_v

        self.Q: T["d_in", "d_out_kq"] = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.K: T["d_in", "d_out_kq"] = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.V: T["d_in", "d_out_v"] = nn.Parameter(torch.rand(d_in, d_out_v))

        self.norm = nn.LayerNorm(d_in)

    def forward(
        self,
        x: TensorType["batch", "n", "self.d_in"],  # noqa: F821
    ) -> T["batch", "n", "self.d_out_v"]:  # noqa: F821
        keys: T["batch", ..., "n", "d_out_kq"] = x @ self.K  # noqa: F821
        queries: T["batch", ..., "n", "d_out_kq"] = x @ self.Q  # noqa: F821
        values: T["batch", ..., "n", "d_out_v"] = x @ self.V  # noqa: F821
        keys_T: T["batch", "d_out_kq", "n"] = keys.swapaxes(-1, -2)  # noqa: F821

        attn_scores: T["batch", "n", "n"] = (  # noqa: F821
            queries @ keys_T
        )
        attn_weights: T["batch", "n", "n"] = torch.softmax(  # noqa: F821
            attn_scores / self.d_out_kq**0.5, dim=-1
        )

        context_vec: T["batch", "n", "d_out_v"] = attn_weights @ values  # noqa: F821
        return context_vec


class TokenDenoiser(nn.Module):
    def __init__(
        self,
        max_T: int,
        d_embed: int,
        n_tokens: int,
        d_postembed: int,
        d_kq: int,
        d_hidden: int,
        n_blocks: int,
    ):
        super().__init__()
        self.max_T = max_T
        self.d_embed = d_embed
        self.n_tokens = n_tokens
        self.d_postembed = d_postembed
        self.d_kq = d_kq
        self.n_blocks = n_blocks

        self.te = TimeEmbedding(
            max_T=max_T,
            d_embed=d_embed,
            d_output=d_postembed,
        )
        self.pe = PositionalEncoding(
            n_tokens=n_tokens,
            d_output=d_postembed,
        )
        self.blocks = nn.Sequential(
            *(
                [
                    SelfAttention(d_in=d_postembed, d_out_kq=d_kq, d_out_v=d_hidden)
                    for _ in range(n_blocks)
                ]
                + [
                    nn.Linear(in_features=d_hidden, out_features=n_tokens),
                    nn.ReLU(),
                ]
            )
        )

    def forward(self, x: T["b", "n_tokens"]) -> T["b", "n_tokens", "d_embed"]:  # noqa: F821
        te = self.te(x)
        pe = self.pe(x)
        xe = te + pe
        return self.blocks(xe).swapaxes(-1, -2)
