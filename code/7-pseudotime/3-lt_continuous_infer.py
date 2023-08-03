#%%
import IPython
if IPython.get_ipython() is not None:
    IPython.get_ipython().magic('load_ext autoreload')
    IPython.get_ipython().magic('autoreload 2')

import os
import sys
import ast
import gzip
import torch
import pickle
import numpy as np
import pandas as pd
import tqdm.auto as tqdm
import chromatinhd as chd

#%%
print('Start of 3-lt_continuous_infer.py')

if len(sys.argv) > 2 and sys.argv[2] == 'external':
    locals().update(ast.literal_eval(sys.argv[1]))
    external = True
else:
    dataset_name_sub = "MV2"
    dataset_name = "myeloid"
    nbins = (128, 64, 32, )
    model_type = 'sigmoid'
    external = False

folder_root = chd.get_output()
folder_data_preproc = folder_root / "data" / "hspc"
models_dir = folder_data_preproc / "models"
fragment_dir = folder_data_preproc / f"{dataset_name_sub}_fragments_{dataset_name}"
df_latent_file = folder_data_preproc / f"{dataset_name_sub}_latent_time_{dataset_name}.csv"

#%%
promoter_name, window = "10k10k", np.array([-10000, 10000])
promoters = pd.read_csv(folder_data_preproc / f"{dataset_name_sub}_promoters_{promoter_name}.csv", index_col = 0)

fragments = chd.data.Fragments(fragment_dir / promoter_name)
fragments.window = window
fragments.create_cut_data()

folds = pd.read_pickle(fragment_dir / promoter_name / "folds.pkl")

# torch works by default in float32
df_latent = pd.read_csv(df_latent_file, index_col = 0)
latent = torch.tensor(df_latent['latent_time'].values.astype(np.float32))

#%%
pseudocoordinates = torch.linspace(0, 1, 1000)
latent_times = torch.linspace(0, 1, 101)
latent_times = torch.flip(latent_times, [0])

#%%
model_pattern = f"{dataset_name_sub}_{dataset_name}_{model_type}_{'_'.join(str(n) for n in nbins)}"
models = sorted([file for file in os.listdir(models_dir) if model_pattern in file])
csv_dir = folder_data_preproc / f"{dataset_name_sub}_LC" / model_pattern
os.makedirs(csv_dir, exist_ok=True)

# csv_dict = {model: {x: None for x in range(promoters.shape[0])} for model in models}
csv_dict = {x: torch.zeros((101, 1000)) for x in range(promoters.shape[0])}

#%%
for model_name in models:
    print(model_name)
    model = pickle.load(open(models_dir / model_name, "rb"))

    for gene_oi in range(promoters.shape[0]):
        print(gene_oi)
        gene_name = promoters.index[gene_oi]
        likelihoods = model.evaluate_pseudo(pseudocoordinates, latent_times, gene_oi)

        # overall = likelihoods.overall
        # overall = torch.exp(overall)
        # overall = torch.reshape(overall, (101, 1000))

        # height = likelihoods.height
        # height = torch.exp(height)
        # height = torch.reshape(height, (101, 1000))

        probs = likelihoods.total
        probs = torch.exp(probs)
        probs = torch.reshape(probs, (101, 1000))

        csv_dict[gene_oi] = csv_dict[gene_oi] + probs

for gene, tensor in csv_dict.items():
    gene_name = promoters.index[gene]
    filename = f"{csv_dir}/{gene_name}.csv.gz"
    tensor = tensor / len(models)

    with gzip.open(filename, 'wt') as file:
        np.savetxt(file, tensor, delimiter=',')

print('End of 3-lt_continuous_infer.py')

#%%
# transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")

# gene_name = 'MPO'
# gene_ix = transcriptome.gene_ix(gene_name) # double check if this is the correct gene_ix

# #%%
# cumulative_sum_list = [0] + [sum(nbins[:i+1]) for i in range(len(nbins))]

# fig, axs = plt.subplots(3, 1, figsize=(8, 10))  # create a 3x1 grid of subplots

# for i in range(len(nbins)):
#     x = cumulative_sum_list[i]
#     y = cumulative_sum_list[i+1]
#     print(x, y)
#     heatmap = axs[i].pcolor(model.decoder.delta_height_slope.data[:, x:y], cmap=plt.cm.RdBu)

# fig.colorbar(heatmap, ax=axs)  # add a colorbar for each subplot
# plt.savefig(folder_data_preproc / "plots" / ("test" + ".pdf"))


# #%%
# cumulative_sum_list = [0] + [sum(nbins[:i+1]) for i in range(len(nbins))]

# fig, axs = plt.subplots(3, 1, figsize=(8, 10))  # create a 3x1 grid of subplots

# for i in range(len(nbins)):
#     x = cumulative_sum_list[i]
#     y = cumulative_sum_list[i+1]
#     print(x, y)
#     heatmap = axs[i].pcolor(torch.abs(model.decoder.delta_height_slope.data[:, x:y]), cmap=plt.cm.Blues)

# fig.colorbar(heatmap, ax=axs)  # add a colorbar for each subplot
# plt.savefig(folder_data_preproc / "plots" / ("test2" + ".pdf"))

# #%%
# cumulative_sum_list = [0] + [sum(nbins[:i+1]) for i in range(len(nbins))]
# for i in range(len(nbins)):
#     x = cumulative_sum_list[i]
#     y = cumulative_sum_list[i+1]
#     print(x, y)
#     plt.figure()
#     plt.plot(torch.abs(model.decoder.delta_height_slope.data[:, x:y]).mean(0))
#     plt.savefig(folder_data_preproc / "plots" / ("delta_height_slope_" + str(nbins[i]) + ".pdf"))
# plt.show() 

# #%%
# trainer.trace.plot()
# pd.DataFrame(trainer.trace.train_steps).groupby("epoch").mean()["loss"].plot()
# pd.DataFrame(trainer.trace.validation_steps).groupby("epoch").mean()["loss"].plot()

# %%
