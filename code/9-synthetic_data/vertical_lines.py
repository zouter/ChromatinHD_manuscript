#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# range for x and y values for NF
x_min, x_max, y_min, y_max = 0, 1, 0, 1

# biological range of x values
p_min, p_max = -10000, 10000

def plot_cutsites(df, gene, n_fragments):
    fig, ax = plt.subplots(figsize=(15, 15))

    ax.scatter(df['x'], df['y'], s=1, marker='s', color='black')
    ax.set_title(f"{gene} (cut sites = {2 * n_fragments})", fontsize=14)
    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Latent Time', fontsize=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_facecolor('white')

def generate_data(p_min, p_max, x_min, x_max, y_min, y_max, *args):
    data = []

    for arg in args:
        n_values, shift_density, noise, shift_slope, p = arg

        # convert to NF range
        x = (p - p_min) / (p_max - p_min) * (x_max - x_min) + x_min

        # create y values
        y_distribution = np.linspace(y_min, y_max, n_values) ** shift_density

        # create x values
        x_distribution = np.full_like(y_distribution, x)

        # add noise to x
        x_distribution = x_distribution + np.random.normal(0, noise, n_values)

        # add shift to x
        x_distribution = x_distribution + np.linspace(x_min, x_max, n_values) * shift_slope

        # column bind x and y values
        data.append(np.column_stack((x_distribution, y_distribution)))

    return data

# parameters for generating data
args_list = [
    [1000, 0.5, 0.01, 0.001, 0], # n_values, shift_density, noise, shift_slope, p
    [500, 0.7, 0.02, 0.002, 4000],
    [2000, 0.3, 0.015, 0.003, -5000],
]

data_list = generate_data(p_min, p_max, x_min, x_max, y_min, y_max, *args_list)

data = np.concatenate(data_list, axis=0)
dataframe = pd.DataFrame(data, columns=['x', 'y'])
# x = genomic position
# y = latent time
# use this to create fragments dir

plot_cutsites(dataframe, 'gene', len(dataframe))