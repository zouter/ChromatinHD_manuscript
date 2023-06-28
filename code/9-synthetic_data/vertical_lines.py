#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# NF range of x values
x_min = 0
x_max = 1

# range of y values
y_min = 0
y_max = 1

# biological range of x values
p_min = -10000
p_max = 10000

# select arbitrary positions
p1 = 0
p2 = 2000
p3 = -2000

# convert to NF range
x1 = (p1 - p_min) / (p_max - p_min) * (x_max - x_min) + x_min
x2 = (p2 - p_min) / (p_max - p_min) * (x_max - x_min) + x_min
x3 = (p3 - p_min) / (p_max - p_min) * (x_max - x_min) + x_min

# %%
# uniform distribution
y1_distribution = np.linspace(x_min, x_max, 1000)
# not uniform distribution
y2_distribution = np.linspace(x_min, x_max, 1000) ** 2
# not uniform distribution in opposite direction
y3_distribution = np.linspace(x_min, x_max, 1000) ** 0.5

# %%
# transform y1_distribution to 2D arry
d1 = np.column_stack((np.full_like(y1_distribution, x1), y1_distribution))
d2 = np.column_stack((np.full_like(y2_distribution, x2), y2_distribution))
d3 = np.column_stack((np.full_like(y3_distribution, x3), y3_distribution))

# add noise to x
d1[:, 0] = d1[:, 0] + np.random.normal(0, 0.01, d1.shape[0])
d2[:, 0] = d2[:, 0] + np.random.normal(0, 0.01, d2.shape[0])
d3[:, 0] = d3[:, 0] + np.random.normal(0, 0.01, d3.shape[0])

# %%
data = np.concatenate((d1, d2, d3), axis=0)
dataframe = pd.DataFrame(data, columns=['x', 'y'])

# %%
def plot_cutsites(df, gene, n_fragments):
    fig, ax = plt.subplots(figsize=(15, 15))

    ax.scatter(df['x'], df['y'], s=1, marker='s', color='black')
    ax.set_title(f"{gene} (cut sites = {2 * n_fragments})", fontsize=14)
    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Latent Time', fontsize=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_facecolor('white')

plot_cutsites(dataframe, 'gene', len(dataframe))
# %%
