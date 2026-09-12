"""Functions completed by students.

Search for the string 'TODO'. The rest of the repository should not need editing.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from torch import nn
import torch.nn.functional as F


LATENT_DIM = 2


class Autoencoder(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28)),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        # TODO 1: encode x into z, then decode z into x_hat.
        # Return BOTH x_hat and z.
        raise NotImplementedError("TODO 1")


def fit_gmm(z: np.ndarray, n_components: int = 10, seed: int = 0) -> GaussianMixture:
    """Fit a full-covariance Gaussian mixture to 2D latent codes."""
    # TODO 2: create a GaussianMixture with `n_components` components and
    # random_state=seed, fit it to z, and return the fitted object.
    raise NotImplementedError("TODO 2")


def sample_gmm(gmm: GaussianMixture, n_samples: int) -> np.ndarray:
    """Draw latent samples from a fitted sklearn GaussianMixture."""
    # TODO 3: use gmm.sample(...) and return ONLY the sampled latent vectors.
    raise NotImplementedError("TODO 3")


class VAE(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_body = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(64, latent_dim)
        self.logvar_head = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28)),
        )

    def encode(self, x: torch.Tensor):
        h = self.encoder_body(x)

        # TODO 4: compute and return the two Gaussian parameters
        # mu(x) and logvar(x) = log(sigma(x)^2).
        raise NotImplementedError("TODO 4")

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # TODO 5: implement
        #     std = exp(0.5 * logvar)
        #     eps ~ N(0, I)
        #     z = mu + std * eps
        raise NotImplementedError("TODO 5")

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z


def vae_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
):
    """Return (total_loss, reconstruction_loss, kl_loss).

    Each term is averaged over the batch. The reconstruction error is summed
    over pixels for each image, matching a Gaussian decoder up to constants.
    """

    # TODO 6: reconstruction loss.
    # Compute the squared error, SUM over pixels for each image, then MEAN
    # over the batch.
    raise NotImplementedError("TODO 6")

    # TODO 7: KL(q(z|x) || N(0,I)).
    # Use the closed-form expression given in the lab statement, SUM over
    # latent dimensions for each image, then MEAN over the batch.
    # Finally set total = reconstruction + beta * kl and return all 3 values.
    raise NotImplementedError("TODO 7")
