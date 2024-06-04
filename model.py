import math

import torch
from torch import nn
import torch.nn.functional as F
from torchtyping import TensorType as T


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


class PositionalEmbedder(nn.Module):
    def __init__(self, num_possible_tokens: int, d_embed: int):
        super().__init__()
        self.num_possible_tokens = num_possible_tokens
        self.d_embed = d_embed
        emb: T["num_possible_tokens", "d_embed"] = positional_embeding_matrix(
            num_possible_tokens, d_embed
        )
        self.register_buffer("emb", emb)

    def forward(
        self, tokens: T["b", "l", torch.long]
    ) -> T["b", "l", "self.d_embed", torch.float]:  # noqa: F821
        return self.emb[tokens]


class Denoiser(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        len_sentence: int,
        d_embed: int,
        num_timestamps: int,
        model: nn.Module,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.len_sentence = len_sentence
        self.d_embed = d_embed

        self.token_embedder = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_embed
        )
        self.time_embedder = PositionalEmbedder(
            num_possible_tokens=num_timestamps, d_embed=d_embed
        )
        self.model = model

    def forward(self, x: T["b", "l", torch.long], t: T["b"]) -> T["b", "l", "v"]:  # noqa: F821
        time: T["b", 1, "self.d_embed"] = self.time_embedder(t).unsqueeze(1)  # noqa: F821
        tokens: T["b", "l", "self.d_embed"] = self.token_embedder(x)  # noqa: F821
        return self.model(time + tokens)


class DownsampleBlock(nn.Module):
    def __init__(self, d_in: int, dropout: float = 0.1):
        super().__init__()
        kernel_size = self.kernel_size = 4
        self.d_in = d_in
        stride = 2
        d_out = self.d_out = d_in // stride
        padding = 2 * stride + kernel_size + 1
        self.layer = nn.Sequential(
            nn.Conv1d(
                in_channels=d_in,
                out_channels=d_out,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(d_out),
        )

    def forward(self, x):
        return self.layer(x)


class UpsampleBlock(nn.Module):
    def __init__(self, d_in: int, dropout: float = 0.1):
        super().__init__()
        self.d_in = d_in
        d_out = d_in * 2
        self.dropout = dropout

        self.us = nn.Upsample(scale_factor=2)
        self.nn = nn.Sequential(
            nn.Conv1d(d_out, d_out, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(d_out),
        )

    def forward(self, x):
        x = self.us(x.swapaxes(-1, -2)).swapaxes(-1, -2)
        return self.nn(x)


class ConvText(nn.Module):
    def __init__(self, d_embed: int = 1024):
        super().__init__()
        self.d_embed = d_embed
        assert d_embed % 4 == 0

        self.ds1 = DownsampleBlock(d_in=d_embed)
        self.ds2 = DownsampleBlock(d_in=d_embed // 2)
        self.us1 = UpsampleBlock(d_in=d_embed // 4)
        self.us2 = UpsampleBlock(d_in=d_embed // 2)

    def forward(self, x: T["b", "d_embed", "l"]) -> T["b", "d_embed", "l"]:  # noqa: F821
        x = x.swapaxes(-1, -2)
        ds1 = self.ds1(x)
        ds2 = self.ds2(ds1)
        us1 = self.us1(ds2)  #  + ds1
        us2 = self.us2(us1)
        return us2.swapaxes(-1, -2)


class MLPText(nn.Module):
    def __init__(self, d_embed: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.d_embed = d_embed
        d_hidden = d_embed // 4

        self.nn = nn.Sequential(
            nn.Linear(in_features=d_embed, out_features=d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_hidden),
            nn.Linear(in_features=d_hidden, out_features=d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_hidden),
            nn.Linear(in_features=d_hidden, out_features=d_embed),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.nn(x)
