#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# range for x and y values for NF
x_min, x_max, y_min, y_max = 0, 1, 0, 1

# biological range of x values
p_min, p_max = -10000, 10000

# number of cells
n_cells = 6037

def plot_cutsites(df, gene, n_fragments):
    fig, ax = plt.subplots(figsize=(15, 15))

    ax.scatter(df['x'], df['y'], s=1, marker='s', color='black')
    ax.set_title(f"{gene} (cut sites = {2 * n_fragments})", fontsize=14)
    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Latent Time', fontsize=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_facecolor('white')

def generate_lt(n_cells, n_cells_x, power):
    # create cell ids
    cell_ids = np.arange(n_cells)
    # shift density
    prob1 = cell_ids ** abs(power)
    # reverse if power is negative
    prob1 = prob1[::-1] if power < 0 else prob1
    # normalize
    prob2 = (prob1 / np.sum(prob1)) * n_cells_x
    # clip
    prob2[prob2 > 0.99] = 0.99
    # create binary vector
    prob3 = np.array([np.random.choice([0, 1], p=[1 - probability, probability]) for probability in prob2])
    # multiply cell ids by binary vector
    prob4 = cell_ids * prob3
    # select cell ids
    prob5 = prob4[prob4 > 0]
    # convert to latent time
    prob6 = prob5 / (n_cells-1)

    return prob6

def generate_gene(p_min, p_max, x_min, x_max, n_cells, *args):
    data = []

    for index, arg in enumerate(args):
        n_values, shift_density, noise, shift_slope, p, fragment_length = arg

        # convert to NF range
        x = (p - p_min) / (p_max - p_min) * (x_max - x_min) + x_min

        # create y values
        y_distribution = generate_lt(n_cells, n_values, shift_density)

        # create x values
        x_distribution = np.full_like(y_distribution, x)

        # add noise to x
        x_distribution = x_distribution + np.random.normal(0, noise, len(x_distribution))

        # add shift to x
        x_distribution = x_distribution + np.linspace(x_min, x_max, len(x_distribution)) * shift_slope

        # add noise to fragment_length
        fragment_length = np.random.normal(fragment_length, 50, len(x_distribution))

        # convert to int
        fragment_length = fragment_length.astype(int)
        print(fragment_length)

        # if smaller than 10, set to 10
        fragment_length[fragment_length < 10] = 10

        # convert fragment_length to NF range
        fragment_length = fragment_length / (p_max - p_min) * (x_max - x_min)

        # add fragment_length to x_distribution
        xf_distribution = x_distribution + fragment_length

        # column bind x and y values
        arr = np.column_stack((x_distribution, xf_distribution, y_distribution))

        # add index
        arr = np.column_stack((np.full_like(arr[:, 0], index), arr))

        # append to data
        data.append(arr)
    
    # col stack
    data = np.concatenate(data, axis=0)

    return data

def generate_genes(gene_parameters):
    genes_data = []

    for index, gene_params in enumerate(gene_parameters):
        gene_data = generate_gene(p_min, p_max, x_min, x_max, n_cells, *gene_params)
        # add index to gene_data
        gene_data = np.column_stack((np.full_like(gene_data[:, 0], index), gene_data))
        genes_data.extend(gene_data)

    return genes_data

# Define the list of gene parameters for generating multiple genes
gene_parameters_list = [
    [
        [1000, 0.5, 0.01, 0.001, 0, 150],  # n_values, shift_density, noise, shift_slope, p, fragment_length
        [500, 0.7, 0.02, 0.002, 4000, 150], 
        [2000, 0.3, 0.015, 0.003, -5000, 150] 
    ],
    [
        [800, 0.6, 0.012, 0.0015, 100, 150], 
        [1200, 0.8, 0.018, 0.0025, -2000, 150], 
    ],
    # ...
    # combinatorially generate data for many genes
]

genes_data_list = generate_genes(gene_parameters_list)


#%%
dataframe = pd.DataFrame(genes_data_list, columns=['gene_ix', 'gene_element', 'cut_start', 'cut_end', 'lt'])
# convert lt to cell_ix
dataframe['cell_ix'] = dataframe['lt'] * (n_cells-1)
# convert NF range to genomic range
dataframe['cut_start'] = dataframe['cut_start'] * (p_max - p_min) + p_min
dataframe['cut_end'] = dataframe['cut_end'] * (p_max - p_min) + p_min

mapping = dataframe[['gene_ix', 'cell_ix']].values
coordinates = dataframe[['cut_start', 'cut_end']].values

#%%
gene_parameters = [
    [1000, 0.5, 0.01, 0.001, 0, 150], # n_values, shift_density, noise, shift_slope, p, fragment_length
    [500, 0.7, 0.02, 0.002, 4000, 150],
    [2000, 0.3, 0.015, 0.003, -5000, 150],
]

data_list = generate_gene(p_min, p_max, x_min, x_max, y_min, y_max, *gene_parameters)
dataframe = pd.DataFrame(data_list, columns=['x1', 'x2', 'y'])

# melt dataframe
dataframe = pd.melt(dataframe, id_vars=['y'], value_name='x')

plot_cutsites(dataframe, 'gene', len(dataframe))

# %%
import pickle

file_path = '/home/vifernan/projects/ChromatinHD_manuscript/output/data/hspc_backup/fragments_myeloid/10k10k/'

coordinates = pickle.load(open(file_path + 'coordinates.pkl', 'rb')) # start x end
mapping = pickle.load(open(file_path + 'mapping.pkl', 'rb')) # cell x gene

# %%
