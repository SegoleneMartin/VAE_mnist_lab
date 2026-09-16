"""Check the Python environment. This script does not download MNIST."""
from __future__ import annotations

import sys


def main() -> None:
    try:
        import torch
        import torchvision
        import sklearn
        import matplotlib
        import numpy
    except Exception as exc:
        print("Python packages: FAILED")
        print(exc)
        sys.exit(1)

    print(f"PyTorch: OK ({torch.__version__})")
    print(f"torchvision: OK ({torchvision.__version__})")
    print(f"scikit-learn: OK ({sklearn.__version__})")
    if torch.cuda.is_available():
        print(f"CUDA: OK ({torch.cuda.get_device_name(0)})")
    else:
        print("CUDA: not detected")
        print("CPU mode is available: use --cpu when running the lab scripts.")
    print("Environment: OK")


if __name__ == "__main__":
    main()
