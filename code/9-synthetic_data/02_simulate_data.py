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
# order of parameters: n_values, shift_y_density, noise, shift_slope, p, fragment_length
gene_parameters_list = pickle.load(open('gene_parameters_list.pickle', 'rb'))

#%%
# create promoter file
promoters = pd.DataFrame(index=np.arange(len(gene_parameters_list)))
promoters.index = promoters.index.map(lambda x: f'gene{x}')
promoters.index.name = 'gene'
promoters.to_csv(folder_data_preproc / ("promoters_10k10k_simulated.csv"))

# Generate the data
genes_data_list = generate_genes(gene_parameters_list)

#%%
# Convert to dataframe
dataframe = pd.DataFrame(genes_data_list, columns=['gene_ix', 'gene_element', 'cut_start', 'cut_end', 'lt'])

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

for gene_id in dataframe['gene_ix'].unique():
    print(gene_id)
    dataframe_sub = dataframe[dataframe['gene_ix'] == gene_id]
    print(len(dataframe_sub))
    # melt dataframe
    df_melt = pd.melt(dataframe_sub, id_vars=['gene_ix', 'gene_element', 'lt'], value_name='x')
    print(len(df_melt))
    # rename columns x and lt
    df_melt = df_melt.rename(columns={'lt': 'y'})
    # plot
    plot_cutsites(df_melt, str(int(df_melt['gene_ix'][0])), len(df_melt))

    if gene_id == 2:
        break

#%%
# convert NF range to genomic range
dataframe['cut_start'] = dataframe['cut_start'] * (p_max - p_min) + p_min
dataframe['cut_end'] = dataframe['cut_end'] * (p_max - p_min) + p_min

# def plot_cutsites(df, gene, n_fragments):
#     fig, ax = plt.subplots(figsize=(15, 15))
#     ax.scatter(df['x'], df['y'], s=1, marker='s', color='black')
#     ax.set_title(f"{gene} (cut sites = {2 * n_fragments})", fontsize=14)
#     ax.set_xlabel('Position', fontsize=12)
#     ax.set_ylabel('Latent Time', fontsize=12)
#     ax.set_xlim([-10000, 10000])
#     ax.set_ylim([0, 1])
#     ax.set_facecolor('white')
#     fig.show()

# for gene_id in dataframe['gene_ix'].unique():
#     print(gene_id)
#     dataframe_sub = dataframe[dataframe['gene_ix'] == gene_id]
#     print(len(dataframe_sub))
#     # melt dataframe
#     df_melt = pd.melt(dataframe_sub, id_vars=['gene_ix', 'gene_element', 'lt'], value_name='x')
#     print(len(df_melt))
#     # rename columns x and lt
#     df_melt = df_melt.rename(columns={'lt': 'y'})
#     # plot
#     plot_cutsites(df_melt, str(int(df_melt['gene_ix'][0])), len(df_melt))

#     if gene_id == 2:
#         break

del genes_data_list

#%%
dataframe['cell_ix'] = dataframe['lt'] * (n_cells-1)

id_exist = set(np.round(sorted(set(dataframe['cell_ix']))).astype(int))
missing_cell_ix = np.array(list(set(np.arange(n_cells)) - id_exist))

df_missing_cell_ix = dataframe[:len(missing_cell_ix)]
df_missing_cell_ix['cell_ix'] = missing_cell_ix

#%%
dataframe = pd.concat([df_missing_cell_ix, dataframe])

#%%
# convert to tensor for fragments dir
coordinates = torch.tensor(dataframe[['cut_start', 'cut_end']].values, dtype = torch.int64)
mapping = torch.tensor(dataframe[['cell_ix', 'gene_ix']].values, dtype = torch.int64)

#%%
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
var.index = var.index.map(lambda x: f'gene{x}')
var["ix"] = np.arange(n_genes)
var.index.name = 'gene'
fragments.var = var

# create obs
obs = pd.DataFrame(index = np.arange(n_cells))
obs.index = obs.index.map(lambda x: f'cell{x}')
obs["ix"] = np.arange(n_cells)
obs.index.name = 'cell'
fragments.obs = obs

# store mapping
fragments.mapping = mapping

# store coordinates
fragments.coordinates = coordinates

# create cellxgene_indptr
fragments.create_cellxgene_indptr()

folds = []
for i in range(5):
    # randomly select 20% of cells
    cells_validation = np.random.choice(np.arange(n_cells), size=int(n_cells * 0.2), replace=False)
    # get the remaining cells
    cells_train = np.setdiff1d(np.arange(n_cells), cells_validation)
    # append to folds
    folds.append({'cells_train': cells_train, 'cells_validation': cells_validation})

# create folds.pkl
pickle.dump(folds, open(file_path / 'folds.pkl', 'wb'))

# plot histogram for each item in folds
for index, fold in enumerate(folds):
    plt.figure(figsize=(5, 5))
    plt.hist(latent_time.iloc[fold['cells_train']], bins=100, alpha=0.5, label='train')
    plt.hist(latent_time.iloc[fold['cells_validation']], bins=100, alpha=0.5, label='validation')
    plt.title(f'Fold {index}')
    plt.legend()
    plt.show()

print('End of script')

# %%
# check other dir for comparison
# var2 = pd.read_csv(file_path_old / 'var.tsv', sep='\t', index_col=0)
# obs2 = pd.read_csv(file_path_old / 'obs.tsv', sep='\t', index_col=0)
# mapping2 = pickle.load(open(file_path_old / 'mapping.pkl', 'rb')) # cell x gene
# coordinates2 = pickle.load(open(file_path_old / 'coordinates.pkl', 'rb')) # start x end
# cellxgene_indptr2 = pickle.load(open(file_path_old / 'cellxgene_indptr.pkl', 'rb'))
# folds2 = pickle.load(open(file_path_old / 'folds.pkl', 'rb'))
# bin_counts = np.bincount(mapping2[:, 0])
