import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import pandas as pd
from diffusion_model import DiffusionModel, NoiseModel
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('../data_train_log_return.csv', header=None).drop(columns=[0])
scaler = StandardScaler().fit(data.values)

n_steps = 100
beta_1 = 0.0001
beta_t = 0.02
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = NoiseModel(n_steps)
model.load_state_dict(torch.load('diffusion_model.pt'))

diffuser = DiffusionModel(model.eval(), n_steps, beta_1, beta_t, device)

test_z = torch.Tensor(pd.read_csv('input_noise_diffusion.csv', header=None, index_col=0).values)
print(test_z.shape)
with torch.no_grad():
    batch = torch.Tensor(test_z)
    sample, all_samples = diffuser.denoise(torch.Tensor(test_z).to(device), n_steps)

df = pd.DataFrame(scaler.inverse_transform(sample.cpu().numpy()))
df.to_csv('test_set_diffusion.csv', header=None)