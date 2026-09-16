"""Part I - Autoencoder and a learned GMM prior on MNIST.

Complete the TODOs, then run for example:
    python autoencoder.py --gpu --sampling-distribution normal

If no GPU is available, use --cpu instead.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from torch import nn
from torch.optim import Adam

from utils import (
    DEFAULT_OUTPUT_DIR,
    get_loaders,
    save_image_grid,
    save_latent_scatter,
    save_reconstructions,
    seed_everything,
)

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

    def forward(self, x: torch.Tensor):
        # TODO 1: encode x into z, decode z into x_hat, and return (x_hat, z).
        raise NotImplementedError("TODO 1")


def reconstruction_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # TODO 2: squared reconstruction error.
    # Sum over ALL pixels of each image, then average over the batch.
    raise NotImplementedError("TODO 2")


def train_autoencoder(model, loader, device, epochs: int, lr: float) -> None:
    optimizer = Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0
        for x, _ in loader:
            x = x.to(device)

            # TODO 3: write one standard PyTorch optimization step:
            #   1. reset gradients
            #   2. forward pass
            #   3. compute reconstruction loss
            #   4. backward pass
            #   5. optimizer step
            raise NotImplementedError("TODO 3")

            running_loss += loss.item() * x.shape[0]
            n_seen += x.shape[0]

        print(f"AE epoch {epoch:02d}/{epochs} | reconstruction loss {running_loss / n_seen:.3f}")


def collect_latents(model, loader, device, max_points: int = 20_000):
    model.eval()
    latents, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            # TODO 4: call the encoder explicitly to obtain the latent codes.
            z = ...
            latents.append(z.cpu())
            labels.append(y)
            if sum(len(v) for v in latents) >= max_points:
                break
    z = torch.cat(latents)[:max_points].numpy()
    y = torch.cat(labels)[:max_points].numpy()
    return z, y


def main(args) -> None:
    seed_everything(args.seed)
    if args.gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("--gpu was requested, but CUDA is not available. Use --cpu instead.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_loaders(
        batch_size=args.batch_size,
        limit_train=2_000 if args.quick else 20_000,
        limit_test=1_000 if args.quick else 5_000,
    )
    epochs = 1 if args.quick else args.epochs

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint = outdir / "ae.pt"

    if args.sampling_distribution == "normal":
        # First run: train the AE and try a hand-chosen N(0,I) sampling distribution.
        model = Autoencoder().to(device)
        train_autoencoder(model, train_loader, device, epochs=epochs, lr=args.lr)
        torch.save(model.state_dict(), checkpoint)

        x_test, _ = next(iter(test_loader))
        x_test = x_test.to(device)
        model.eval()
        with torch.no_grad():
            x_hat, _ = model(x_test)
            z_normal = torch.randn(args.n_samples, LATENT_DIM, device=device)
            normal_samples = model.decoder(z_normal)

        save_reconstructions(x_test, x_hat, outdir / "ae_reconstructions.png", "Autoencoder reconstructions")
        z_test, y_test = collect_latents(model, test_loader, device, max_points=5_000)
        save_latent_scatter(z_test, y_test, outdir / "ae_latent.png", "Autoencoder latent space")
        save_image_grid(normal_samples, outdir / "ae_normal_samples.png", "AE: z sampled from N(0,I)")
        print(f"Saved AE checkpoint and normal-prior figures in {outdir}")
        return

    # Second run: keep the same trained AE and learn a GMM on its latent codes.
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Missing {checkpoint}. First run autoencoder.py with --sampling-distribution normal."
        )
    model = Autoencoder().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    z_train, _ = collect_latents(model, train_loader, device)
    z_test, y_test = collect_latents(model, test_loader, device, max_points=5_000)

    # TODO 5: learn a latent distribution from the encoded training data.
    # 1. Create a 10-component full-covariance GaussianMixture with random_state=args.seed.
    # 2. Fit it to z_train.
    # 3. Draw args.n_samples latent vectors using gmm.sample(...).
    # 4. Convert the vectors to a float32 tensor on `device` and decode them (you can use torch.as_tensor)
    gmm = ...
    z_samples = ...
    gmm_samples = ...

    # sklearn may return samples grouped by component. Shuffle the displayed
    # images so the grid does not artificially look sorted by mixture component.
    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(len(gmm_samples))
    if isinstance(gmm_samples, torch.Tensor):
        gmm_samples = gmm_samples[torch.as_tensor(permutation, device=gmm_samples.device)]

    save_latent_scatter(
        z_test, y_test, outdir / "ae_latent_gmm.png", "AE latent space with fitted GMM", gmm=gmm
    )
    save_image_grid(gmm_samples, outdir / "ae_gmm_samples.png", "AE: z sampled from fitted GMM")
    print(f"Saved GMM figures in {outdir}")

def parse_args():
    parser = argparse.ArgumentParser(description="MNIST autoencoder + GMM lab")
    device_group = parser.add_mutually_exclusive_group(required=True)
    device_group.add_argument("--cpu", action="store_true", help="run on the CPU")
    device_group.add_argument("--gpu", action="store_true", help="run on one CUDA GPU")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument(
        "--sampling-distribution",
        choices=["normal", "gmm"],
        default="normal",
        help="normal: train AE and sample N(0,I); gmm: load trained AE and fit/sample a GMM",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="1 epoch on a small subset")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
