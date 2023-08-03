#%%
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import chromatinhd as chd

# Set device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set folder paths
folder_root = chd.get_output()
folder_data = folder_root / "data"
dataset_name = "hspc"
folder_data_preproc = folder_data / dataset_name
csv_dir = folder_data_preproc / "evaluate_pseudo_continuous_tensors"
files = sorted(os.listdir(csv_dir))

#%%
probsx_all = []
for x in files[:1]:
    probsx = np.loadtxt(csv_dir / x, delimiter=',')
    probsx = torch.from_numpy(probsx).float()
    probsx_all.append(probsx)

probsx_all = torch.stack(probsx_all, dim=2).float().to(device)
probsx_all = probsx_all[:, :200, :]

#%%
n_latent, n_cutsites, n_genes = probsx_all.shape
probsx_all_reshaped = probsx_all.view(n_latent, -1)
lt = torch.linspace(0, 1, n_latent).float().reshape(-1, 1).to(device)

regression_models = []
regression_slopes = []

# Define the batch size
batch_size = 100
print(f"{batch_size=}")

# Define the Linear Regression model
class LinearRegression(nn.Module):
    def __init__(self, input_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

# Iterate over each batch of columns
for batch_start in range(0, probsx_all_reshaped.size(1), batch_size):
    batch_end = min(batch_start + batch_size, probsx_all_reshaped.size(1))
    batch_columns = probsx_all_reshaped[:, batch_start:batch_end].reshape(-1, batch_end - batch_start, 1)

    # Create a list to store models and slopes for the current batch
    batch_models = []
    batch_slopes = []

    idx_list = list(range(batch_end - batch_start))

    # Iterate over each column in the current batch
    # Create a linear regression model for the current column
    model = LinearRegression(input_size=batch_end-batch_start).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.2)
    loss_history = []

    # Perform linear regression
    for epoch in range(500):
        y_data = batch_columns[:, idx_list, :]

        # Forward pass
        outputs = model(lt)
        loss = criterion(outputs, y_data)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Append the trained model to the batch_models list
    batch_models.append(model)
    batch_slopes.extend(model.linear.weight)

    # Extend the regression_models and regression_slopes lists with the batch results
    regression_models.extend(batch_models)
    regression_slopes.extend(batch_slopes)
    print(f"{len(regression_models)=}")

# %%
