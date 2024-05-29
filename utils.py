import polars as pl
import torch


class OutdatedError(Exception):
    pass


def single_offset_expr(offset: int, context_length: int, start_or_end_token: str):
    name = "sequence"
    colname = f"ch{context_length + 1 - offset:0{context_length}}"
    assert len(start_or_end_token) == 1
    x = start_or_end_token

    return colname, (
        pl.concat_str(
            [pl.lit(x) for _ in range(offset)]
            + [pl.col(name)]
            + [pl.lit(x) for _ in range(context_length + 1 - offset)]
        )
        .alias("sequence")
        .str.reverse()
        .str.slice(offset=context_length)
        .str.reverse()
        .alias(colname)
        .str.split(by="")
    )


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
                pl.concat_str(
                    pl.lit(self.start_token),
                    pl.col(colname),
                    pl.lit(self.end_token),
                )
                .str.pad_end(self.max_len, self.pad_token)
                .alias("tokenized")
                .str.split(by="")
                .cast(pl.List(pl.String))
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

    def into_ngrams(self, df: pl.LazyFrame, context_length: int) -> pl.DataFrame:
        raise OutdatedError
        offset_adding_expr = reversed(
            [
                single_offset_expr(
                    offset=i,
                    context_length=context_length,
                    start_or_end_token=self.start_or_end_token,
                )
                for i in range(context_length + 1)
            ]
        )
        colnames, expressions = list(zip(*offset_adding_expr))
        return (
            df.with_columns(*expressions)
            .select(colnames)
            .explode(colnames)
            .filter(*(pl.col(colname).is_in(self.alphabet) for colname in colnames))
        )

    def ngrams2torch(
        self,
        ngrams: pl.LazyFrame | pl.DataFrame,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise OutdatedError
        # ensure non-lazyness
        ngrams = ngrams.lazy().collect()
        stoi = self.stoi

        renames = {
            colname: pl.col(colname).replace(stoi).cast(pl.Int8)
            for colname in ngrams.columns
        }
        ngrams = ngrams.with_columns(**renames)

        last_col = sorted(ngrams.columns)[-1]
        ys = ngrams.get_column(last_col).to_torch()

        xs_colnames = sorted(ngrams.columns)[:-1]
        xs = ngrams.select(xs_colnames).to_torch()

        return xs, ys

    def ngrams2torch_ohe(
        self,
        ngrams: pl.LazyFrame | pl.DataFrame,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise OutdatedError
        stoi = self.stoi
        ngrams = ngrams.lazy().collect()
        last_col = sorted(ngrams.columns)[-1]
        ys = torch.tensor(
            [
                stoi[char]
                for char in ngrams.lazy().collect().get_column(last_col).to_list()
            ]
        ).int()

        # ensure that all valid amino acids are present, and the order is appropriate
        xs_colnames = sorted(ngrams.columns)[:-1]
        dummy_xs_colnames = [
            f"{colname}_{aa}" for colname in xs_colnames for aa in self.alphabet
        ]

        ngrams = ngrams.to_dummies()
        for colname in dummy_xs_colnames:
            if colname not in ngrams.columns:
                ngrams = ngrams.with_columns(pl.lit(0, dtype=pl.Int8).alias(colname))

        xs = ngrams.select(dummy_xs_colnames).to_torch().float()

        return xs, ys
