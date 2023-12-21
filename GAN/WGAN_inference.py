import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from WGAN_model import W_Generator

data = pd.read_csv('../data_train_log_return.csv', header=None).drop(columns=[0])
scaler = StandardScaler().fit(data.values)

# Parameters
latent_dim = 64  # latent dim
data_dim = 4    # data dimension
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# initialize
generator = W_Generator(latent_dim, data_dim).to(device)

generator.load_state_dict(torch.load('WGAN_Generator.pt'))

z = torch.Tensor(pd.read_csv('input_noise_GAN.csv', header=None, index_col=0).values).to(device)

with torch.no_grad():
    sample = generator(z)

df = pd.DataFrame(scaler.inverse_transform(sample.cpu().numpy()))
df.to_csv('test_set_WGAN.csv', header=None)