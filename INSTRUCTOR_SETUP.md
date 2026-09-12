# Instructor setup notes

This file is not needed by students.

## Recommended timing

- 0–25 min: SSH / clone / Conda / `check_install.py`
- 25–45 min: AE
- 45–60 min: GMM on AE latent codes
- 60–95 min: VAE TODOs + training
- 95–110 min: latent grid + interpolation
- 110–120 min: report answers / figure selection

The beta experiment is optional.

## Dataset

For a classroom session, pre-download MNIST in a shared readable directory instead of relying on internet access during the lab.

Example on a machine with internet access:

```bash
python check_install.py --data-dir /shared/path/mnist
```

Students can then use:

```bash
python check_install.py --data-dir /shared/path/mnist --no-download
python run_lab.py ae --data-dir /shared/path/mnist --no-download
```

## Default compute

The scripts use 20,000 training images and 5,000 test images by default. This is enough for the qualitative experiment and makes training short on a small GPU.

If the cluster is fast, you can use all MNIST images with:

```bash
python run_lab.py ae --limit-train 60000 --limit-test 10000
python run_lab.py vae --limit-train 60000 --limit-test 10000
```

## Student TODOs

There are exactly seven conceptual TODOs in `student.py`:

1. AE forward pass
2. fit GMM
3. sample GMM
4. VAE Gaussian encoder heads
5. reparameterization trick
6. reconstruction loss
7. KL + total VAE loss

The training loops, plotting code, data loading, checkpointing, device handling, and GMM visualization are deliberately provided.
