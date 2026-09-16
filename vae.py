"""Part II - Variational Autoencoder on MNIST.

Complete the TODOs, then run for example:
    python vae.py --gpu

If no GPU is available, use --cpu instead.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

from utils import (
    DEFAULT_OUTPUT_DIR,
    get_loaders,
    save_image_grid,
    save_interpolation,
    save_latent_grid,
    save_latent_scatter,
    save_reconstructions,
    save_training_curves,
    seed_everything,
)

LATENT_DIM = 2


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

        # TODO 1: create TWO linear heads from the 64-dimensional hidden
        # representation to the latent space: one for mu and one for log(sigma^2).
        self.mu_head = ...
        self.logvar_head = ...

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
        # TODO 2: use the two heads to return mu(x) and logvar(x).
        raise NotImplementedError("TODO 2")

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # TODO 3: implement z = mu + sigma * epsilon with epsilon ~ N(0,I).
        # Remember: logvar = log(sigma^2).
        raise NotImplementedError("TODO 3")

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar, z


def vae_loss(x_hat, x, mu, logvar, beta: float = 1.0):
    # TODO 4: reconstruction term.
    reconstruction = ...

    # TODO 5: KL(q(z|x) || N(0,I)).
    # Use the closed-form expression given in the lab statement and implement it here.
    kl = ...

    total = reconstruction + beta * kl
    return total, reconstruction, kl


def train_vae(model, loader, device, epochs: int, lr: float, beta: float):
    optimizer = Adam(model.parameters(), lr=lr)
    history = {"total": [], "reconstruction": [], "KL": []}

    for epoch in range(1, epochs + 1):
        model.train()
        sums = {key: 0.0 for key in history}
        n_seen = 0
        for x, _ in loader:
            x = x.to(device)

            # TODO 6: write the optimization step, as for the autoencoder.
            # Use vae_loss(...) and keep the variables loss, reconstruction, kl.
            raise NotImplementedError("TODO 6")

            bs = x.shape[0]
            sums["total"] += loss.item() * bs
            sums["reconstruction"] += reconstruction.item() * bs
            sums["KL"] += kl.item() * bs
            n_seen += bs

        for key in history:
            history[key].append(sums[key] / n_seen)
        print(
            f"VAE epoch {epoch:02d}/{epochs} | total {history['total'][-1]:.3f} | "
            f"reconstruction {history['reconstruction'][-1]:.3f} | KL {history['KL'][-1]:.3f}"
        )
    return history


def collect_latent_means(model, loader, device, max_points: int = 5_000):
    model.eval()
    means, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            mu, _ = model.encode(x.to(device))
            means.append(mu.cpu())
            labels.append(y)
            if sum(len(v) for v in means) >= max_points:
                break
    return torch.cat(means)[:max_points].numpy(), torch.cat(labels)[:max_points].numpy()


def tagged(filename: str, tag: str) -> str:
    if not tag:
        return filename
    p = Path(filename)
    return f"{p.stem}_{tag}{p.suffix}"


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

    model = VAE().to(device)
    history = train_vae(model, train_loader, device, epochs=epochs, lr=args.lr, beta=args.beta)
    torch.save(model.state_dict(), outdir / tagged("vae.pt", args.tag))

    x_test, _ = next(iter(test_loader))
    x_test = x_test.to(device)
    model.eval()
    with torch.no_grad():
        x_hat, _, _, _ = model(x_test)
        z = torch.randn(args.n_samples, LATENT_DIM, device=device)
        samples = model.decoder(z)

    save_reconstructions(
        x_test,
        x_hat,
        outdir / tagged("vae_reconstructions.png", args.tag),
        f"VAE reconstructions (beta={args.beta:g})",
    )
    z_mean, y = collect_latent_means(model, test_loader, device)
    save_latent_scatter(
        z_mean,
        y,
        outdir / tagged("vae_latent.png", args.tag),
        f"VAE latent means (beta={args.beta:g})",
    )
    save_image_grid(
        samples,
        outdir / tagged("vae_prior_samples.png", args.tag),
        f"VAE: z sampled from N(0,I), beta={args.beta:g}",
    )
    save_training_curves(history, outdir / tagged("vae_training_curves.png", args.tag))
    save_latent_grid(model, device, outdir / tagged("vae_latent_grid.png", args.tag))
    save_interpolation(model, test_loader, device, outdir / tagged("vae_interpolation.png", args.tag))

    print(f"Saved figures and checkpoint in {outdir}")


def parse_args():
    parser = argparse.ArgumentParser(description="MNIST VAE lab")
    device_group = parser.add_mutually_exclusive_group(required=True)
    device_group.add_argument("--cpu", action="store_true", help="run on the CPU")
    device_group.add_argument("--gpu", action="store_true", help="run on one CUDA GPU")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--tag", default="")
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="1 epoch on a small subset")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
