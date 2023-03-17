#%%
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np

def generate_mixture_of_gaussians(num_of_points):
    n = num_of_points // 2
    gaussian1 = np.random.normal(loc=-1, scale=0.25, size=(n, ))
    gaussian2 = np.random.normal(loc=0.5, scale=0.5, size=(num_of_points-n, ))
    return np.concatenate([gaussian1, gaussian2])


class NumpyDataset(data.Dataset):
    def __init__(self, array):
        super().__init__()
        self.array = array

    def __len__(self):
        return len(self.array)
    
    def __getitem__(self, index):
        return self.array[index]

n_train, n_test = 2000, 1000
train_data = generate_mixture_of_gaussians(n_train)
test_data = generate_mixture_of_gaussians(n_test)

train_loader = data.DataLoader(NumpyDataset(train_data), batch_size=128, shuffle=True)
test_loader = data.DataLoader(NumpyDataset(test_data), batch_size=128, shuffle=False)

class Flow1d(nn.Module):
    def __init__(self, n_components):
        super(Flow1d, self).__init__()
        self.mus = nn.Parameter(torch.randn(n_components), requires_grad=True)
        self.log_sigmas = nn.Parameter(torch.zeros(n_components), requires_grad=True)
        self.weight_logits = nn.Parameter(torch.ones(n_components), requires_grad=True)
    
    def forward(self, x):
        x = x.view(-1,1)
        weights = self.weight_logits.softmax(dim=0).view(1,-1)
        distribution = torch.distributions.Normal(self.mus, self.log_sigmas.exp())
        z = (distribution.cdf(x) * weights).sum(dim=1)
        dz_by_dx = (distribution.log_prob(x).exp() * weights).sum(dim=1)
        return z, dz_by_dx
    
def loss_function(target_distribution, z, dz_by_dx):
    log_likelihood = target_distribution.log_prob(z) + dz_by_dx.log()
    return -log_likelihood.mean()

def train(model, train_loader, optimizer, target_distribution):
    model.train()
    for x in train_loader:
        z, dz_by_dx = model(x)
        loss = loss_function(target_distribution, z, dz_by_dx)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def eval_loss(model, data_loader, target_distribution):
    model.eval()
    total_loss = 0
    for x in data_loader:
        z, dz_by_dx = model(x)
        loss = loss_function(target_distribution, z, dz_by_dx)
        total_loss += loss * x.size(0)
    return (total_loss / len(data_loader.dataset)).item()


# %%
