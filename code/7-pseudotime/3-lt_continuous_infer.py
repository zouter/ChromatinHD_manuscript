#%%
import IPython
if IPython.get_ipython() is not None:
    IPython.get_ipython().magic('load_ext autoreload')
    IPython.get_ipython().magic('autoreload 2')

import os
import torch
import torch_scatter
import pickle
import pathlib
import tempfile
import numpy as np
import pandas as pd
import scanpy as sc
import tqdm.auto as tqdm

import chromatinhd as chd
import chromatinhd.loaders.fragments
import chromatinhd.models.likelihood_pseudotime.v1 as likelihood_model

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
sns.set_style('ticks')

#%%
folder_root = chd.get_output()
folder_data_preproc = folder_root / "data" / "hspc"

#%%
promoter_name, window = "10k10k", np.array([-10000, 10000])
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)

fragments = chd.data.Fragments(folder_data_preproc / "fragments_myeloid" / promoter_name)
fragments.window = window
fragments.create_cut_data()

folds = pd.read_pickle(folder_data_preproc / "fragments_myeloid" / promoter_name / "folds.pkl")

# torch works by default in float32
df_latent = pd.read_csv(folder_data_preproc / "MV2_latent_time_myeloid.csv", index_col = 0)
latent = torch.tensor(df_latent['latent_time'].values.astype(np.float32))

#%%
pseudocoordinates = torch.linspace(0, 1, 1000)
latent_times = torch.linspace(0, 1, 101)
latent_times = torch.flip(latent_times, [0])

# #%%
nbins = (256, )
nbins = (128, 64, 32, )
nbins = (128, )
model_pattern = f"3-lt_continuous_{'_'.join(str(n) for n in nbins)}_fold_"
models = sorted([file for file in os.listdir('.') if model_pattern in file])
models_name = [x.replace('3-lt_continuous_', '').replace('.pkl', '') for x in models]

#%%
for index in range(len(models)):
    print(index)

    model = pickle.load(open(f"./{models[index]}", "rb"))

    csv_dir = folder_data_preproc / f"likelihood_continuous_{models_name[index]}"
    print(csv_dir)
    os.makedirs(csv_dir, exist_ok=True)

    for gene_oi in range(promoters.shape[0]):
        print(gene_oi)
        gene_name = promoters.index[gene_oi]
        probs = model.evaluate_pseudo(pseudocoordinates, latent_times, gene_oi)
        probsx = torch.exp(probs)
        probsx = torch.reshape(probsx, (101, 1000))
        np.savetxt(f"{csv_dir}/{gene_name}.csv", probsx, delimiter=',')

print('Inference complete, csv files saved')
# potentially change this to a 3D tensor (cut_site x latent_time x gene) and save the tensor directly

#%%
# latent_oi = torch.tensor([0.0])
# probs0 = model.evaluate_pseudo(pseudocoordinates, latent_oi, gene_oi)

# latent_oi = torch.tensor([0.1])
# probs1 = model.evaluate_pseudo(pseudocoordinates, latent_oi, gene_oi)

# latent_oi = torch.tensor([0.2])
# probs2 = model.evaluate_pseudo(pseudocoordinates, latent_oi, gene_oi)

# latent_oi = torch.tensor([0.0, 0.1, 0.2])
# probs3 = model.evaluate_pseudo(pseudocoordinates, latent_oi, gene_oi)

# probs_all = torch.cat((probs0, probs1, probs2), dim=0)
# sum(probs_all != probs3) # should sum to 0

# #%%
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
