import torch
import torch.nn as nn

class W_Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(W_Generator, self).__init__()
        w = 8
        self.fc = nn.Sequential(
            nn.Linear(input_dim, w),
            nn.ReLU(),
            nn.Linear(w, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

class W_Critic(nn.Module):
    def __init__(self, input_dim):
        super(W_Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(0),
            nn.Linear(64, 32),
            nn.ReLU(0),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.fc(x)