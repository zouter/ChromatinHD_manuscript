#%%
import torch
import shutil
import pickle
import numpy as np
import pandas as pd
import chromatinhd as chd
import matplotlib.pyplot as plt

folder_root = chd.get_output()
folder_data_preproc = folder_root / "data" / "hspc_backup"
df_latent = pd.read_csv(folder_data_preproc / "MV2_latent_time_myeloid.csv", index_col = 0)

# range for x and y values for NF
x_min, x_max, y_min, y_max = 0, 1, 0, 1

# biological range of x values
p_min, p_max = -10000, 10000

# number of cells
n_cells = len(df_latent)

# create latent time
latent_time = np.arange(n_cells) / (n_cells-1)
latent_time = pd.DataFrame(latent_time, columns=['latent_time'])
latent_time.index = 'c' + latent_time.index.astype(str)
latent_time.index.name = 'cell'
latent_time.to_csv(folder_data_preproc / "MV2_latent_time_simulated.csv")

def plot_cutsites(df, gene, n_fragments):
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.scatter(df['x'], df['y'], s=1, marker='s', color='black')
    ax.set_title(f"{gene} (cut sites = {2 * n_fragments})", fontsize=14)
    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Latent Time', fontsize=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_facecolor('white')
    fig.show()

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

        # create y values
        y_distribution = generate_lt(n_cells, n_values, shift_density)

        # create x values
        x_distribution = np.full_like(y_distribution, p)

        # add noise to x
        x_distribution = x_distribution + np.random.normal(0, noise, len(x_distribution))

        # add shift to x
        x_distribution = x_distribution + np.linspace(p_min, p_max, len(x_distribution)) * shift_slope

        # convert to int
        x_distribution = x_distribution.astype(int)

        # convert to NF range
        x_distribution = (x_distribution - p_min) / (p_max - p_min) * (x_max - x_min) + x_min

        # add noise to fragment_length
        fragment_length = np.random.normal(fragment_length, 50, len(x_distribution))

        # if smaller than 10, set to 10
        fragment_length[fragment_length < 10] = 10

        # convert to int
        fragment_length = fragment_length.astype(int)

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
        gene_data = np.column_stack((np.full_like(gene_data[:, 0], index), gene_data))
        genes_data.extend(gene_data)

    return genes_data
 
#%%
# Define the list of gene parameters for generating multiple genes
gene_parameters_list = [
    [
        [1000, 0.5, 20, 0.001, 0, 150],  # n_values, shift_density, noise, shift_slope, p, fragment_length
        [500, 0.7, 30, 0.002, 4000, 150], 
        [2000, 0.3, 15, 0.003, -5000, 150] 
    ],
    [
        [800, 0.6, 12, 0.0015, 100, 150], 
        [1200, 0.8, 18, 0.0025, -2000, 150], 
    ],
    # ...
    # combinatorially generate data for many genes
]

# Generate the data
genes_data_list = generate_genes(gene_parameters_list)

# Convert to dataframe
dataframe = pd.DataFrame(genes_data_list, columns=['gene_ix', 'gene_element', 'cut_start', 'cut_end', 'lt'])

#%%
# split dataframe into sub dataframes by columns 'gene_ix'
for x in [dataframe[dataframe['gene_ix'] == i] for i in dataframe['gene_ix'].unique()]:
    # melt dataframe
    df = pd.melt(x, id_vars=['gene_ix', 'gene_element', 'lt'], value_name='x')
    # rename columns x and lt
    df = df.rename(columns={'lt': 'y'})
    # plot
    plot_cutsites(df, str(int(df['gene_ix'][0])), len(df))

#%%
# convert lt to cell_ix
dataframe['cell_ix'] = (dataframe['lt'] * (n_cells-1)).astype(int)

# convert NF range to genomic range
dataframe['cut_start'] = dataframe['cut_start'] * (p_max - p_min) + p_min
dataframe['cut_end'] = dataframe['cut_end'] * (p_max - p_min) + p_min

#%%
missing_cell_ix = np.array(list(set(np.arange(n_cells)) - set(dataframe['cell_ix'])))

df_missing_cell_ix = dataframe[:len(missing_cell_ix)]
df_missing_cell_ix['cell_ix'] = missing_cell_ix
dataframe = pd.concat([dataframe, df_missing_cell_ix])

#%%
# convert to tensor for fragments dir
coordinates = torch.tensor(dataframe[['cut_start', 'cut_end']].values, dtype = torch.int64)
mapping = torch.tensor(dataframe[['cell_ix', 'gene_ix']].values, dtype = torch.int64)

# dataframe length unique column gene_ix
n_genes = len(dataframe['gene_ix'].unique())

# Sort `coordinates` and `mapping` according to `mapping`
sorted_idx = torch.argsort((mapping[:, 0] * n_genes + mapping[:, 1]))
mapping = mapping[sorted_idx]
coordinates = coordinates[sorted_idx]

#%%
# create fragments dir
file_path = folder_data_preproc / 'fragments_simulated/10k10k/'
file_path_old = folder_data_preproc / 'fragments_myeloid/10k10k/'

#%%
# create dir
file_path.mkdir(parents=True, exist_ok=True)

# create fragments object
fragments = chd.data.Fragments(file_path)

# create var
var = pd.DataFrame(index = np.arange(n_genes))
var["ix"] = np.arange(n_genes)
var.index.name = 'gene'
fragments.var = var

# create obs
obs = pd.DataFrame(index = np.arange(n_cells))
obs["ix"] = np.arange(n_cells)
obs.index.name = 'cell'
fragments.obs = obs

# store mapping
fragments.mapping = mapping

# store coordinates
fragments.coordinates = coordinates

# create cellxgene_indptr
fragments.create_cellxgene_indptr()

# copy the folds.pkl file from file_path_old to file_path
shutil.copy2(file_path_old / 'folds.pkl', file_path / 'folds.pkl')

# %%
# var2 = pd.read_csv(file_path_old / 'var.tsv', sep='\t', index_col=0)
# obs2 = pd.read_csv(file_path_old / 'obs.tsv', sep='\t', index_col=0)
# mapping2 = pickle.load(open(file_path_old / 'mapping.pkl', 'rb')) # cell x gene
# coordinates2 = pickle.load(open(file_path_old / 'coordinates.pkl', 'rb')) # start x end
# cellxgene_indptr2 = pickle.load(open(file_path_old / 'cellxgene_indptr.pkl', 'rb'))
# folds2 = pickle.load(open(file_path_old / 'folds.pkl', 'rb'))
# bin_counts = np.bincount(mapping2[:, 0])

# %%
