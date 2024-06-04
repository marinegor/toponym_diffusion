from torchtyping import TensorType
import torch
from torch import nn

SequenceBatch = TensorType["batch", "len_sentence", torch.long]
SequenceEmbedding = TensorType["batch", "len_sequence", "d_embeding", torch.float]
Timestamps = TensorType["batch", torch.long]


class Scheduler:
    def __init__(self, T: int, beta_0: float = 1e-4, beta_T: float = 2e-2):
        """Initialize a scheduler.

        Note that we pre-compute self._alphas, self._betas and self._alphabars:
        they are pre-defined for each timepoint, hence we can initialize them once
        and then sample given a timepoint tensor.

        Parameters
        ----------
        T : int
            total number of timepoints to sample
        beta_0 : float, optional
            value when t=0, by default 1e-4
        beta_T : float, optional
            value when t=T, by default 2e-2
        """
        assert beta_0 <= beta_T, "beta_0 is greater than beta_T"
        assert 0 < beta_0 < 1 and 0 < beta_T < 1
        self.T = T
        self.beta_0 = beta_0
        self.beta_T = beta_T

        # we need:
        #  - "noise power" aka variance -- betas
        #  - alphas = 1-betas
        #  - alpha_bars = cumprod(alphas)

        self._betas = torch.linspace(beta_0, beta_T, T + 1)
        self._alphas = 1 - self._betas
        self._alphabars = torch.cumprod(self._alphas, dim=-1)

    def _check_input(self, t: Timestamps):
        """Checks if given timestamps are compatible with the current scheduler

        Parameters
        ----------
        t : Timestamps
            input timestamps
        """
        assert (
            0 <= t.min().item() and t.max().item() <= self.T
        ), (
            f"Timestamps don't match maximum T={self.T}: {t.min()=}, {t.max()=}"
        )  # check that there'll be no indexing error

    def get_betas(self, t: Timestamps) -> TensorType["batch", torch.float]:  # noqa: F821
        """Return $\beta_t$ for all timestamps in `t`.

        Returns
        -------
        TensorType["batch", torch.float]
            Output betas
        """
        self._check_input(t)
        return self._betas[t]

    def get_alphas(self, t: Timestamps) -> TensorType["batch", torch.float]:  # noqa: F821
        """Return $\alpha_t$ for all timestamps in `t`.

        Returns
        -------
        TensorType["batch", torch.float]
            Output alphas
        """
        self._check_input(t)
        return self._alphas[t]

    def get_alphabar(self, t: Timestamps) -> TensorType["batch", torch.float]:  # noqa: F821
        """Return $\alphabar_t$ for all timestamps in `t`.
        This is used later to generate t-th sample in closed form,
        instead of sampling it iteratively.

        Returns
        -------
        TensorType["batch", torch.float]
            Output alphabar parameters.
        """
        self._check_input(t)
        return self._alphabars[t]


class ForwardDiffusionProcess:
    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler

    def sample_t_plus_1(self, x_t: SequenceBatch, t: Timestamps) -> SequenceBatch:
        """Samples next image for all images in batch, given timestamps.

        Timestamps are only needed to get the beta parameters -- in a simplest
        case, they are constant and can safely be omited. However, for more complex
        schedulers, they must be present to sample correctly.

        Here we're using the following equation:

        $$
        q(x_t | x_{t-1}) ~ N(\sqrt(1 - \beta_t)x_{t-1}, \beta_t I)
        $$

        which is essentially a single step of the diffusion process.

        Parameters
        ----------
        x_t : SequenceBatch
            input images
        t : Timestamps
            input timestamps for each image

        Returns
        -------
        SequenceBatch
            transformed images
        """
        beta_t: TensorType["batch", torch.float] = self.scheduler.get_betas(t)  # noqa: F821
        beta_t: TensorType["batch", 1, 1, 1, torch.float] = beta_t.view(  # noqa: F821
            beta_t.shape[0], *[1 for _ in x_t.shape[1:]]
        )  # the .view() is to make pytorch broadcasting work

        mean: SequenceBatch = (1 - beta_t).sqrt() * x_t
        std: SequenceBatch = torch.randn_like(x_t) * beta_t
        return mean + std

    def sample_T(self, x_0: SequenceBatch, t: Timestamps) -> SequenceBatch:
        """Sample a certain timepoint in a closed form

        Parameters
        ----------
        x_0 : SequenceBatch
            starting (unperturbed) image
        t : Timestamps
            timestamps

        Returns
        -------
        SequenceBatch
            output, perturbed image
        """
        alpha_bar_t: TensorType["batch", torch.float] = self.scheduler.get_alphabar(t)  # noqa: F821
        alpha_bar_t: TensorType["batch", 1, 1, 1, torch.float] = alpha_bar_t.view(  # noqa: F821
            alpha_bar_t.shape[0], *[1 for _ in x_0.shape[1:]]
        )

        mean: SequenceBatch = alpha_bar_t.sqrt() * x_0
        std: SequenceBatch = torch.randn_like(x_0) * (1 - alpha_bar_t)

        return mean + std

    def sample(self, model: nn.Module, n: int, max_T: int):
        model.eval()
        with torch.no_grad():
            noise_probs = torch.randint(
                low=0, high=model.alphabet_size, size=(n, model.n_tokens)
            ).float()

            for t in reversed(range(max_T)):
                t = (torch.ones(n) * t).long()
                x = predicted_probs = model(noise_probs, t)

                alpha = self.scheduler.get_alphas(t).reshape(-1, 1, 1)
                alphabar = self.scheduler.get_alphabar(t).reshape(-1, 1, 1)
                beta = self.scheduler.get_betas(t).reshape(-1, 1, 1)
                noise = torch.randn_like(x)

                x = (
                    1
                    / torch.sqrt(alpha)
                    * (x - ((1 - alpha) / (torch.sqrt(1 - alphabar))) * x)
                    + torch.sqrt(beta) * noise
                )

        return predicted_probs
