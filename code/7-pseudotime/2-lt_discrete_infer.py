#%%
import IPython
if IPython.get_ipython() is not None:
    IPython.get_ipython().magic('load_ext autoreload')
    IPython.get_ipython().magic('autoreload 2')

import os
import torch
import torch_scatter
import pickle
import numpy as np
import pandas as pd
import tqdm.auto as tqdm
import chromatinhd as chd

# %%
folder_root = chd.get_output()
folder_data_preproc = folder_root / "data" / "hspc"
dataset_name = "erythroid"
dataset_name_sub = "MV2"
model_type = 'quantile'
nbins = (128, 64, 32, )
models_dir = folder_data_preproc / "models"
fragment_dir = folder_data_preproc / f"{dataset_name_sub}_fragments_{dataset_name}"
df_latent_file = folder_data_preproc / f"{dataset_name_sub}_latent_time_{dataset_name}.csv"

promoter_name, window = "10k10k", np.array([-10000, 10000])
promoters = pd.read_csv(folder_data_preproc / f"{dataset_name_sub}_promoters_{promoter_name}.csv", index_col = 0)

info_genes_cells = pd.read_csv(folder_data_preproc / "info_genes_cells.csv")
lineage = info_genes_cells[f'lin_{dataset_name}'].dropna().tolist()

# here quantiles based on latent time are used
df_latent = pd.read_csv(df_latent_file, index_col = 0)
df_latent['quantile'] = pd.qcut(df_latent['latent_time'], 10, labels = False) + 1
latent = pd.get_dummies(df_latent['quantile'])
latent_torch = torch.from_numpy(latent.values).to(torch.float)

fragments = chd.data.Fragments(fragment_dir / promoter_name)
fragments.window = window
fragments.create_cut_data()

#%%
model_pattern = f"{dataset_name_sub}_{dataset_name}_{model_type}_{'_'.join(str(n) for n in nbins)}"
models = sorted([file for file in os.listdir(models_dir) if model_pattern in file])
csv_dir = folder_data_preproc / f"{dataset_name_sub}_LQ" / model_pattern
os.makedirs(csv_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pseudocoordinates = torch.linspace(0, 1, 1000).to(device)
csv_dict = {x: torch.zeros((len(pseudocoordinates), latent.shape[1])).to(device) for x in range(promoters.shape[0])}

def evaluate_pseudo_quantile(latent, gene_oi, model, device):
    pseudocoordinates = torch.linspace(0, 1, 1000).to(device)
    probs_tensor = torch.zeros((len(pseudocoordinates), latent.shape[1])).to(device)
    for i in range(latent.shape[1]):
        pseudolatent = torch.zeros((len(pseudocoordinates), latent.shape[1])).to(device)
        pseudolatent[:, i] = 1.
        probs_tensor[:, i] = model.evaluate_pseudo(pseudocoordinates.to(device), latent = pseudolatent.to(device), gene_oi = gene_oi)
    probs_tensor = torch.exp(probs_tensor)
    return probs_tensor

# %%
for model_name in models:
    model = pickle.load(open(models_dir / model_name, "rb"))
    model = model.to(device).eval()
    model = model.to(device)

    for gene_oi in range(len(promoters)):
        print(gene_oi)
        gene_id = fragments.var.index[gene_oi]
        probs = evaluate_pseudo_quantile(latent, gene_oi, model, device)
        print(probs)
        csv_dict[gene_oi] = csv_dict[gene_oi] + probs

# %%
for gene, tensor in csv_dict.items():
    gene_name = promoters.index[gene]
    filename = f"{csv_dir}/{gene_name}.csv.gz"
    tensor = tensor / len(models)
    probs_df = pd.DataFrame(tensor.cpu(), index = pseudocoordinates.tolist(), columns = latent.columns)
    probs_df.to_csv(filename, compression='gzip')

print("2-lt_discrete_infer.py done")

# %%
