from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from student import Autoencoder, VAE, fit_gmm, sample_gmm, vae_loss
from utils import (
    collect_ae_latents,
    collect_vae_means,
    get_loaders,
    resolve_device,
    save_image_grid,
    save_interpolation,
    save_latent_grid,
    save_latent_scatter,
    save_reconstructions,
    save_training_curves,
    seed_everything,
)


def tag_name(base: str, tag: str) -> str:
    if not tag:
        return base
    p = Path(base)
    return f"{p.stem}_{tag}{p.suffix}"


def train_ae(model, loader, device, epochs: int, lr: float):
    opt = Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for x, _ in tqdm(loader, desc=f"AE epoch {epoch}/{epochs}", leave=False):
            x = x.to(device)
            opt.zero_grad()
            x_hat, _ = model(x)
            loss = F.mse_loss(x_hat, x, reduction="sum") / x.shape[0]
            loss.backward()
            opt.step()
            total += loss.item() * x.shape[0]
            count += x.shape[0]
        print(f"AE epoch {epoch:02d} | reconstruction loss {total / count:.3f}")


def train_vae(model, loader, device, epochs: int, lr: float, beta: float):
    opt = Adam(model.parameters(), lr=lr)
    history = {"total": [], "reconstruction": [], "KL": []}
    for epoch in range(1, epochs + 1):
        model.train()
        sums = {k: 0.0 for k in history}
        count = 0
        for x, _ in tqdm(loader, desc=f"VAE epoch {epoch}/{epochs}", leave=False):
            x = x.to(device)
            opt.zero_grad()
            x_hat, mu, logvar, _ = model(x)
            loss, rec, kl = vae_loss(x_hat, x, mu, logvar, beta=beta)
            loss.backward()
            opt.step()
            bs = x.shape[0]
            sums["total"] += loss.item() * bs
            sums["reconstruction"] += rec.item() * bs
            sums["KL"] += kl.item() * bs
            count += bs
        for k in history:
            history[k].append(sums[k] / count)
        print(
            f"VAE epoch {epoch:02d} | total {history['total'][-1]:.3f} | "
            f"reconstruction {history['reconstruction'][-1]:.3f} | KL {history['KL'][-1]:.3f}"
        )
    return history


def first_batch(loader, device):
    x, y = next(iter(loader))
    return x.to(device), y


def command_ae(args, train_loader, test_loader, device, outdir):
    model = Autoencoder().to(device)
    train_ae(model, train_loader, device, args.epochs_ae, args.lr)
    ckpt = outdir / tag_name("ae", args.tag)
    torch.save(model.state_dict(), ckpt.with_suffix(".pt"))

    x, _ = first_batch(test_loader, device)
    model.eval()
    with torch.no_grad():
        x_hat, _ = model(x)
        prior_z = torch.randn(64, 2, device=device)
        prior_samples = model.decode(prior_z)

    save_reconstructions(x, x_hat, outdir / tag_name("ae_reconstructions.png", args.tag), "Autoencoder reconstructions")
    z, y = collect_ae_latents(model, test_loader, device)
    save_latent_scatter(z, y, outdir / tag_name("ae_latent.png", args.tag), "Autoencoder latent space")
    save_image_grid(prior_samples, outdir / tag_name("ae_prior_samples.png", args.tag), "AE: samples from z ~ N(0,I)")
    print(f"Saved AE checkpoint and figures in {outdir}")


def load_ae(args, device, outdir):
    ckpt = outdir / tag_name("ae", args.tag)
    path = ckpt.with_suffix(".pt")
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run `python run_lab.py ae` first.")
    model = Autoencoder().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def command_gmm(args, train_loader, test_loader, device, outdir):
    model = load_ae(args, device, outdir)
    z_train, _ = collect_ae_latents(model, train_loader, device, max_points=args.gmm_points)
    gmm = fit_gmm(z_train, n_components=10, seed=args.seed)
    z_test, y_test = collect_ae_latents(model, test_loader, device)
    save_latent_scatter(z_test, y_test, outdir / tag_name("ae_latent_gmm.png", args.tag), "AE latent space with fitted GMM", gmm=gmm)

    z_new = sample_gmm(gmm, 64)
    z_new = torch.as_tensor(z_new, dtype=torch.float32, device=device)
    with torch.no_grad():
        samples = model.decode(z_new)
    save_image_grid(samples, outdir / tag_name("ae_gmm_samples.png", args.tag), "AE: samples from fitted GMM")
    print(f"Saved GMM figures in {outdir}")


def command_vae(args, train_loader, test_loader, device, outdir):
    model = VAE().to(device)
    history = train_vae(model, train_loader, device, args.epochs_vae, args.lr, args.beta)
    ckpt = outdir / tag_name("vae", args.tag)
    torch.save(model.state_dict(), ckpt.with_suffix(".pt"))

    x, _ = first_batch(test_loader, device)
    model.eval()
    with torch.no_grad():
        x_hat, _, _, _ = model(x)
        z = torch.randn(64, 2, device=device)
        samples = model.decode(z)

    save_reconstructions(x, x_hat, outdir / tag_name("vae_reconstructions.png", args.tag), f"VAE reconstructions (beta={args.beta:g})")
    z_mean, y = collect_vae_means(model, test_loader, device)
    save_latent_scatter(z_mean, y, outdir / tag_name("vae_latent.png", args.tag), f"VAE latent means (beta={args.beta:g})")
    save_image_grid(samples, outdir / tag_name("vae_prior_samples.png", args.tag), f"VAE: samples from z ~ N(0,I), beta={args.beta:g}")
    save_training_curves(history, outdir / tag_name("vae_training_curves.png", args.tag))
    print(f"Saved VAE checkpoint and figures in {outdir}")


def load_vae(args, device, outdir):
    ckpt = outdir / tag_name("vae", args.tag)
    path = ckpt.with_suffix(".pt")
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run `python run_lab.py vae` first.")
    model = VAE().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def command_explore(args, test_loader, device, outdir):
    model = load_vae(args, device, outdir)
    save_latent_grid(model, device, outdir / tag_name("vae_latent_grid.png", args.tag))
    save_interpolation(model, test_loader, device, outdir / tag_name("vae_interpolation.png", args.tag))
    print(f"Saved exploration figures in {outdir}")


def parse_args():
    parser = argparse.ArgumentParser(description="MNIST AE/GMM/VAE practical lab")
    parser.add_argument("command", choices=["ae", "gmm", "vae", "explore"])
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs-ae", type=int, default=5)
    parser.add_argument("--epochs-vae", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--tag", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gmm-points", type=int, default=20_000)
    parser.add_argument("--limit-train", type=int, default=20_000)
    parser.add_argument("--limit-test", type=int, default=5_000)
    parser.add_argument("--quick", action="store_true", help="Very small run for debugging only")
    parser.add_argument("--no-download", action="store_true", help="Do not let torchvision download MNIST")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    if args.quick:
        args.limit_train = 2_000
        args.limit_test = 1_000
        args.epochs_ae = min(args.epochs_ae, 1)
        args.epochs_vae = min(args.epochs_vae, 1)
        args.gmm_points = min(args.gmm_points, 2_000)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = get_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        limit_train=args.limit_train,
        limit_test=args.limit_test,
        download=not args.no_download,
    )

    if args.command == "ae":
        command_ae(args, train_loader, test_loader, device, outdir)
    elif args.command == "gmm":
        command_gmm(args, train_loader, test_loader, device, outdir)
    elif args.command == "vae":
        command_vae(args, train_loader, test_loader, device, outdir)
    elif args.command == "explore":
        command_explore(args, test_loader, device, outdir)


if __name__ == "__main__":
    main()
