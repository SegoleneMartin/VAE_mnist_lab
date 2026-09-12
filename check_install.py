from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    try:
        import torch
        import torchvision
        import sklearn
        import matplotlib
        import numpy as np
        from torchvision import datasets, transforms
    except Exception as exc:
        print("Python packages: FAILED")
        print(exc)
        sys.exit(1)

    print(f"PyTorch: OK ({torch.__version__})")
    if torch.cuda.is_available():
        print(f"CUDA: OK ({torch.cuda.get_device_name(0)})")
    else:
        print("CUDA: not detected (CPU is still usable for debugging)")

    try:
        ds = datasets.MNIST(
            args.data_dir,
            train=True,
            transform=transforms.ToTensor(),
            download=not args.no_download,
        )
        x, y = ds[0]
        assert tuple(x.shape) == (1, 28, 28)
        print(f"MNIST: OK ({len(ds)} training images, data dir: {Path(args.data_dir).resolve()})")
    except Exception as exc:
        print("MNIST: FAILED")
        print(exc)
        print("If downloads are disabled on the cluster, ask the instructor for the shared MNIST path.")
        sys.exit(2)

    print("Everything is ready!")


if __name__ == "__main__":
    main()
