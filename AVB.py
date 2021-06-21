import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import os
import pandas as pd
from numpy import loadtxt
from torch.autograd import Variable
import math
from sklearn.feature_selection import mutual_info_regression
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf


#导入数据
#导入数据
Lorenz = np.loadtxt('D:/Lorenz5W.txt')
Lorenz = np.array(Lorenz.data)
Data = Lorenz

mb_size = 1000
Z_dim = 2
X_dim = 3
X_length = Data.shape[0]
h_dim = 512
c = 0
lr = 3e-4
eps_dim = 4

# Rand_Weight = np.random.randn(Z_dim, X_dim)
# Entangle_Data = np.dot(Data, Rand_Weight)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.Q_state = torch.nn.Sequential(
            nn.Linear(X_dim, h_dim),
            nn.Tanh()
        )
        self.linear_mu = torch.nn.Linear(h_dim, Z_dim)
        self.linear_var = torch.nn.Linear(h_dim, Z_dim)
        self.P_generate = torch.nn.Sequential(
            nn.Linear(Z_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, X_dim)
        )
        self.T_discriminator = torch.nn.Sequential(
            torch.nn.Linear(X_dim + Z_dim, h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim, 1)
        )

    def forward(self, X):
        # ======= Q(z|X) =========================
        h = self.Q_state(X)
        z_mu = self.linear_mu(h)
        z_var = self.linear_var(h)
        # ============ sample z ===========
        eps = torch.randn(mb_size, Z_dim)
        z_sample = z_mu + torch.exp(z_var / 2) * eps
        # ==========  P(X_sample | z) ============
        X_sample = self.P_generate(z_sample)
        return X_sample, z_mu, z_var
    def parameter(self):
        return

def My_loss(X, X_sample, z_mu, z_var):
    recon_loss = F.mse_loss(X_sample, X, reduction='sum') / mb_size
    kl_loss = torch.mean(
        0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1. - z_var, 1))
    Loss = recon_loss + kl_loss
    return Loss

def log(x):
    return torch.log(x + 1e-8)

Q = torch.nn.Sequential(
    torch.nn.Linear(X_dim + eps_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, Z_dim)
)

T = torch.nn.Sequential(
    torch.nn.Linear(X_dim + Z_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, 1)
)

model1 = Model()
solver = optim.Adam(model1.parameters(), lr=lr)
for it in range(50000):
    i_batch = it % (math.floor(X_length / mb_size))
    X = Data[mb_size*i_batch:mb_size*(i_batch+1), :]
    # print(X)
    X = torch.tensor(X, dtype=torch.float32)
    X_sample, z_mu, z_var = model1.forward(X)
    eps = torch.randn(mb_size, Z_dim)
    z = torch.randn(mb_size, Z_dim)
    z_sample = z_mu + torch.exp(z_var / 2) * eps
    z_sample = z_sample.detach().numpy()
    z_sample = torch.from_numpy(z_sample)
    X = X.detach().numpy()
    X = torch.from_numpy(X)
    T_sample = T(torch.cat([X, z_sample], 1))
    disc = torch.mean(-T_sample)
    loglike = -F.mse_loss(X_sample, X, reduction='sum') / mb_size
    elbo = -(disc + loglike)
    # compute loss
    #z_sample = Q(torch.cat([X, eps], 1))
    T_q = F.sigmoid(T(torch.cat([X, z_sample], 1)))
    T_prior = F.sigmoid(T(torch.cat([X, z], 1)))

    T_loss = -torch.mean(log(T_q) + log(1. - T_prior))
    # Backward
    solver.zero_grad()
    T_loss.backward()
    # Update
    elbo.backward()
    solver.step()

    if it % 1000 == 0:
        print('Iter-{}; Loss: {:.4}'.format(it, T_loss))
        samples, z_mu, z_var = model1.forward(X)
        print(samples.shape)
        samples = samples.detach().numpy()
        #X = X.detach().numpy()
        #eps = torch.randn(mb_size, Z_dim)
        #z_sample = z_mu + torch.exp(z_var / 2) * eps
        #z_sample = z_sample.detach().numpy()
