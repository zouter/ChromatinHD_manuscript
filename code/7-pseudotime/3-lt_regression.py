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

probsx_all = []
for x in files:
    probsx = np.loadtxt(csv_dir / x, delimiter=',')
    probsx = torch.from_numpy(probsx).float()
    probsx_all.append(probsx)

probsx_all = torch.stack(probsx_all, dim=2).float().to(device)
print(f"{probsx_all.shape=}")

n_latent, n_cutsites, n_genes = probsx_all.shape
probsx_all_reshaped = probsx_all.view(n_latent, -1)
lt = torch.linspace(0, 1, n_latent).float().reshape(-1, 1).to(device)

regression_models = []
regression_slopes = []

# Define the Linear Regression model
class LinearRegression(nn.Module):
    def __init__(self, input_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

# Iterate over each column in probsx_all_reshaped
for i in range(probsx_all_reshaped.size(1)):
    print(i)
    # Create a linear regression model for the current column
    model = LinearRegression(input_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.2)
    loss_history = []

    # Perform linear regression
    for epoch in range(500):
        y_data = probsx_all_reshaped[:, i].reshape(-1, 1)

        # Forward pass
        outputs = model(lt)
        loss = criterion(outputs, y_data)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Append the loss to the history list
    #     loss_history.append(loss.item())

    # plt.plot(loss_history)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Loss Curve')
    # plt.show()

    # Add the trained model to the list of regression models
    # regression_models.append(model)
    regression_models.append(model.linear.weight.item())

regression_slopes = torch.tensor(regression_slopes)
regression_slopes = torch.reshape(regression_slopes, (1, 1000, 877))
torch.save(regression_slopes, csv_dir / 'regression_slopes.pt')
print('Saved regression slopes')

# %%
for x in range(probsx_all_reshaped.shape[-1]):
    model = regression_models[x]

    x_data = lt.cpu().numpy()
    y_data = probsx_all_reshaped[:, x].reshape(-1, 1).cpu().numpy()
    y_model = model(lt).detach().cpu().numpy()

    plt.figure()
    plt.scatter(x_data, y_data, color='blue', label='Data')
    plt.plot(x_data, y_model, color='red', label='Model')
    plt.xlabel('lt')
    plt.ylabel(f'probsx_all_reshaped[:, {x}]')
    plt.title(f'Scatter Plot for Column {x+1}')
    plt.legend()
    plt.show()

    if x == 5:
        break

# %%
