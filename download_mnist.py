"""Download MNIST into the data/ directory of this repository."""
from pathlib import Path
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

print(f"Downloading MNIST to: {DATA_DIR}")
transform = transforms.ToTensor()
train = datasets.MNIST(DATA_DIR, train=True, transform=transform, download=True)
test = datasets.MNIST(DATA_DIR, train=False, transform=transform, download=True)
print(f"MNIST: OK ({len(train)} train / {len(test)} test images)")
