import math

import torch
from torch import nn
from torchtyping import TensorType

T = TensorType


def init_weights(m: nn.Module):
    """Initialize weights for all linear weights in a model with xavier normal initialization

    Parameters
    ----------
    m : nn.Module
        input model
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)


def positional_embeding_matrix(
    position_id: int, d_out: int
) -> T["position_id", "out_dim"]:  # noqa: F821
    """Generates a pre-computed tensor with positional embeddings.

    Returns
    -------
    torch.Tensor
        embedings tensor; `i` position contains vector for `if token is at position "i"`
    """
    assert d_out % 2 == 0, f"d_out should be even, got {d_out=}"

    position: T["position_id", 1] = torch.arange(position_id).unsqueeze(-1)
    div_term: T["d_out // 2"] = (
        torch.arange(0, d_out, 2) * (-math.log(10_000.0) / d_out)
    ).exp()
    pe = torch.zeros(position_id, d_out)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class PositionalEncoding(nn.Module):
    """Add positional embedings to the input"""

    def __init__(self, d_embed: int, max_L: int):
        """Initialize the layer and register a `self.te` buffer

        Parameters
        ----------
        d_embed : int
            length of the time vector for each timepoint; second dim of `self.te`
        max_L : int
            maximal time allowed by the layer; first dim of `self.te`
        """
        super().__init__()
        self.max_L = max_L
        self.d_embed = d_embed

        pe = positional_embeding_matrix(position_id=max_L, d_out=d_embed)
        self.register_buffer("pe", pe)

    def forward(self, x: T["b", "l", "d"]) -> T["b", "l", "d"]:  # noqa: F821
        """Add positional embeding to the input

        Returns
        -------
        torch.Tensor
            shape (batch, l, d) where 'l' is number of tokens in each sequence,
            and 'd' is size of each token's embeding
        """

        # pe: T["b", "d"] = self.pe[x]  # noqa: F821
        pe: T["b", "d"] = self.pe[:, : x.size(1)]  # noqa: F821
        return pe


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
        max_L: int,
        d_embed: int,
        d_kq: int,
        d_hidden: int,
        n_blocks: int = 3,
    ):
        super().__init__()

        self.te = PositionalEncoding(
            max_L=max_T,
            d_embed=d_embed,
        )
        self.pe = PositionalEncoding(
            max_L=max_L,
            d_embed=d_embed,
        )
        self.blocks = nn.Sequential(
            *[
                SelfAttention(
                    d_in=d_embed,
                    d_out_kq=d_kq,
                    d_out_v=d_hidden,
                )
                for _ in range(n_blocks)
            ],
            *[
                nn.Linear(d_hidden, max_L),
                nn.ReLU(),
            ],
        )

    def forward(
        self,
        x: T["b", "l"],  # noqa: F821
        t: T["b"],  # noqa: F821
    ) -> T["b", "l", "n_tokens"]:  # noqa: F821
        te: T["b", 1, "d_embed"] = self.te(t).unsqueeze(1)  # noqa: F821
        pe: T["b", "l", "d_embed"] = self.pe(x)  # noqa: F821
        xe = te + pe
        return self.blocks(xe)  # .swapaxes(-1, -2)
