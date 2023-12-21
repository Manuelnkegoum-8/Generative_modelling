# Generative modeling project
![Python](https://img.shields.io/badge/Python-3.10.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-%23EE4C2C)



The main objective of our project was , Given a vector of input noise vector $\mathbf{Z} \in \mathbb{R}^N$, build a generative model $G_{\theta}$, parametrized by $\theta$, that can simulate realistic samples $G_{\theta}(\mathbf{Z}) = \tilde{\mathbf{X}} \in \mathbb{R}^d$ that are similar to the real negative financial returns $\mathbf{X} \in \mathbb{R}^d$.

We built different models : Normalizing flows, Diffusion models and Generative adversarial Networks.This repo  contains 3 directories (one for each model implementation).Each directory contains:
- The model architecture written in pytorch
- The weights of the corresponding trained model

*For a detailed explanation of each implementation please refer to the report.*
## Getting started 🚀
**We provide necessary informations to run our code and generating samples.
By default we generated 410 samples and the model used is the realnvp which proved us the best results.**
### Realnvp
```bash
python Inference.py
```
### GAN
```bash
python Inference.py --model gan
```
### Diffusion
```bash
python Inference.py --model diffusion
```

## Authors  🧑‍💻
- Haocheng Liu 
-  Manuel Nkegoum