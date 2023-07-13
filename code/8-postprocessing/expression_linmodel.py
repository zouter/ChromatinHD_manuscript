# %%
import IPython
if IPython.get_ipython() is not None:
    IPython.get_ipython().magic('load_ext autoreload')
    IPython.get_ipython().magic('autoreload 2')

import pickle
import numpy as np
import pandas as pd
import chromatinhd as chd
import statsmodels.api as sm

# %%
folder_root = chd.get_output()
folder_data_preproc = folder_root / "data" / "hspc"
dataset_name_sub = "MV2"

#%%
transcriptome = chd.data.Transcriptome(folder_data_preproc / f"{dataset_name_sub}_transcriptome")
adata = transcriptome.adata

# %%
def prepare_data(adata, lineage):
    lt = pd.read_csv(folder_data_preproc / f"{dataset_name_sub}_latent_time_{lineage}.csv", index_col=0)
    matrix = adata[lt.index].X
    x = lt['latent_time'].to_numpy()
    return matrix, x

def linear_model(matrix, x):
    slopes = []
    for i in range(matrix.shape[1]):
        print(i)

        expression = matrix[:, i].toarray().squeeze()
        model = sm.OLS(expression, x)
        results = model.fit()
        slopes.append(results.params[0])

    return slopes

lineages = ["myeloid", "erythroid", "platelet"]
for lineage in lineages:
    print(lineage)
    matrix, x = prepare_data(adata, lineage)
    slopes = linear_model(matrix, x)
    adata.var[f"slope_{lineage}"] = slopes

# %%
dict_lineages = {}
for x in lineages:
    dict_lineages[f"{x}_increasing"] = adata.var.sort_values(by=f'slope_{x}', ascending=False).head(100)['Accession'].to_list()
    dict_lineages[f"{x}_decreasing"] = adata.var.sort_values(by=f'slope_{x}', ascending=True).head(100)['Accession'].to_list()

pickle.dump(dict_lineages, open(folder_data_preproc / f'{dataset_name_sub}_100_genes_high_low_slope.pkl', 'wb'))

# %%
