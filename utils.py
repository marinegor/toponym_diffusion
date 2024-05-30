import polars as pl
import torch


class OutdatedError(Exception):
    pass


class Tokenizer:
    def __init__(
        self,
        alphabet: str,
        max_len: int,
        start_token: str = "<",
        end_token: str = ">",
        pad_token: str = ".",
    ):
        assert len(start_token) == len(end_token) == len(pad_token) == 1
        assert not any((t in alphabet for t in (start_token, end_token, pad_token)))
        assert isinstance(alphabet, str), f"{type(alphabet)=}"
        alphabet = f"{start_token}{end_token}{pad_token}{alphabet}"

        self.alphabet = sorted(list(alphabet))
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.max_len = max_len
        self._stoi = {char: idx for idx, char in enumerate(alphabet)}
        self._itos = {idx: char for idx, char in enumerate(alphabet)}

    @property
    def stoi(self):
        return self._stoi

    @property
    def itos(self):
        return self._itos

    def encode(self, df: pl.DataFrame, colname: str = "sequence") -> torch.Tensor:
        """Returns OHE encodings of input sequence, padded to max length.

        Parameters
        ----------
        df : pl.DataFrame
            input dataframe
        colname : str, optional
            column to encode, by default "sequence"
        """
        return (
            df.with_columns(
                pl.concat_str(  # add start and end tokens
                    pl.lit(self.start_token),
                    pl.col(colname),
                    pl.lit(self.end_token),
                )
                .str.pad_end(self.max_len, self.pad_token)  # pad string to max length
                .alias("tokenized")
                .str.split(by="")
                .list.eval(pl.element().replace(self.stoi, return_dtype=pl.UInt8)), 
            )
            .get_column("tokenized")
            .cast(pl.Array(pl.UInt8, self.max_len))
            .to_torch()
        )

    def decode_raw(self, tensor: torch.Tensor) -> list[str]:
        return ["".join(self.itos[i.item()] for i in row) for row in tensor]

    def decode(self, tensor: torch.Tensor) -> list[str]:
        return [
            "".join(self.itos[i.item()].replace(self.pad_token, "") for i in row)[1:-1]
            for row in tensor
        ]
