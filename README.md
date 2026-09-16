# MNIST VAE practical lab

Read **README.pdf** for the assignment (LaTeX source: `README.tex`).

```bash
conda env create -f environment.yml
conda activate genmodels-vae
python check_install.py
python download_mnist.py
```

Then complete the TODOs in `autoencoder.py` and `vae.py`.
Every training command requires either `--gpu` or `--cpu`.
