# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Preprocess

# %%
from IPython import get_ipython

if get_ipython():
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")

import torch

import pickle

import scanpy as sc

import tqdm.auto as tqdm
import io

# %%
import chromatinhd as chd

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"
# dataset_name = "hspc_cycling"
# dataset_name = "hspc_hsc_cycling"
# dataset_name = "hspc_meg_cycling"
dataset_name = "hspc_gmp_cycling"

folder_data_preproc = folder_data / "hspc" / "MV2"
folder_data_preproc.mkdir(exist_ok = True, parents = True)
genome = "GRCh38"

# %% [markdown]
# ## Preprocess

# %%
dataset_folder = chd.get_output() / "datasets" / dataset_name
dataset_folder.mkdir(exist_ok=True, parents=True)

# %%
adata_original = pickle.load((folder_data_preproc / "adata_annotated.pkl").open("rb"))

# %%
sc.pl.umap(adata_original, color = ["phase", "leiden"], legend_loc = "on data")

# %%
# adata = adata[adata.obs["leiden"].isin(["8", "1", "15", "7", "5", "2"])]
# adata = adata[adata.obs["leiden"].isin(["1", "15", "7", "5", "2"])]
# adata = adata[adata.obs["leiden"].isin(["8"])]
# adata = adata_original[adata_original.obs["leiden"].isin(["19", "12"])]
adata = adata_original[adata_original.obs["leiden"].isin(["2", "5"])]
# paths = {'x': [-3.145282886451484, -3.9202355636878092, -4.001809529712686, -3.8794485806753713, -3.390004784526113, -2.207182277165406, -1.7585254640285861, -1.3914426169166425, -1.3914426169166425, -1.8400994300534623, -2.003247362103215], 'y': [4.046594253086661, 2.886071213816544, 2.0519452793411466, 1.653015484592044, 1.0002212750026027, 0.5287587902991175, 0.5287587902991175, 1.0002212750026027, 1.9431462444095733, 3.611398113360367, 3.828996183223514]}
# paths = {'x': [-3.716980265713167, -2.2078618942529546, -1.840779047141011, -1.7184180981036965, -1.188187318942, -0.9842524038798093, -0.5763825737554275, -0.5763825737554275, -0.7803174888176184, -1.4736962000290674, -1.8815660301534491, -2.6565187073897745, -3.5130453506509762, -4.165637078849987, -4.247211044874863, -4.206424061862425], 'y': [4.2279259779726175, 3.0311365937253085, 2.269543349204294, 1.1815529998885588, 0.637557825230691, 0.27489437545877937, -1.3933574934920145, -1.6109555633551615, -1.8648199781954997, -2.1549507380130293, -2.082418048058647, -1.3570911485148234, -0.16030176426751466, 1.1090203099341762, 2.0519452793411466, 3.2850010085656467]}
paths = {'x': [-3.141114001152576, -3.509996976452074, -3.632957968218573, -3.5509839737075737, -2.649270034086579, -2.157426067020582, -1.1737381328885879, -0.7638681603335904, -0.6818941658225909, -1.0507771411220888, -1.501634110932586], 'y': [0.337713352427159, -0.42216330326322976, -0.42216330326322976, -0.7116401244786159, -1.507701382820928, -2.1590242305555467, -2.30376264116324, -2.2313934358593936, -2.0866550252517, -1.1096707536497719, -0.27742489265553666]}
polygon = np.array([paths["x"], paths["y"]]).T
import matplotlib.path as mpath
path = mpath.Path(polygon)
inside = path.contains_points(np.array([adata_original.obsm["X_umap"][:, 0], adata_original.obsm["X_umap"][:, 1]]).T)
adata = adata_original[inside]

# %%
sc.pl.umap(adata, color = ["phase"])

# %%
adata = adata.copy()
adata.X = adata.layers["normalized"]
adata.var = adata.var[[col for col in adata.var.columns if col not in ["dispersion", "dispersions_norm"]]]
sc.pp.highly_variable_genes(adata)

# %%
sc.pl.umap(adata, color = gene_id(adata_original, "SPI1"), use_raw = False, layer = "normalized")

# %%
adata.var.query("symbol == 'MKI67'")


# %%
def gene_id(adata, symbol):
    return adata.var.reset_index().set_index("symbol").loc[symbol, "gene"]
def symbol(adata, gene):
    return adata.var.reset_index().set_index("gene").loc[gene, "symbol"]


# %%
plotdata = pd.DataFrame({
    "expression":sc.get.obs_df(adata, gene_id(adata, "SPI1"), layer = "normalized"),
    "phase":adata.obs["phase"],
})
plotdata.groupby("phase").mean()

# %%
# plotdata["expression"] = (plotdata["expression"] > 0).astype(int)

# %%
sns.boxplot(data = plotdata, x = "phase", y = "expression")

# %% [markdown]
# ### Create transcriptome

# %%
desired_genes = ["SPI1", "GATA1", "KLF1", "FLI1", "RUNX1", "CEBPD", "CEBPA"]
undesired_genes = ["ENSG00000283886"]

# %%
genes_oi_1 = adata.var.sort_values("dispersions_norm").tail(2000).index
genes_oi_2 = adata.var.reset_index().set_index("symbol").loc[desired_genes, "gene"].values

genes_oi = np.concatenate([genes_oi_1, genes_oi_2[~np.isin(genes_oi_2, genes_oi_1)]])
genes_oi = genes_oi[~np.isin(genes_oi, undesired_genes)]
transcriptome = chd.data.transcriptome.Transcriptome.from_adata(adata[:, genes_oi], path=dataset_folder / "transcriptome", overwrite=True)

# %%
adata.X = adata.layers['normalized']
sc.tl.rank_genes_groups(adata, groupby = "phase", use_raw = False)

# %%
diffexp = sc.get.rank_genes_groups_df(adata, group = "G2M")
diffexp["gene"] = adata.var.loc[diffexp["names"], "symbol"].values
diffexp = diffexp.set_index("gene")
# diffexp.loc["MPO"]

# %%
adata2 = adata_original[((adata_original.obs["leiden"] == "0") | (adata_original.obs.index.isin(adata.obs.index))) & (adata_original.obs["phase"] == "G1")].copy()
adata2.X = adata2.layers['normalized']
adata2.obs["group"] = (adata2.obs["leiden"] == "0").astype(int).astype("category")
sc.tl.rank_genes_groups(adata2, groupby = "group", use_raw = False)

diffexp_diff = sc.get.rank_genes_groups_df(adata2, group = None)
diffexp_diff["gene"] = adata2.var.loc[diffexp_diff["names"], "symbol"].values
diffexp_diff = diffexp_diff.query("pvals_adj < 0.05").query("logfoldchanges > 1")

# %%
diffexp.join(diffexp_diff.set_index("gene"), lsuffix = "cc", rsuffix = "diff").query("pvalscc < 0.001").query("scorescc > 3").query("logfoldchangescc > 0.3").sort_values("scoresdiff", ascending = False).head(50)

# %%
adata_original.var.loc[adata_original.var.symbol.str.startswith("PAX")]

# %%
sc.pl.umap(adata_original, color = adata_original.var.index[(adata_original.var["symbol"] == "CDK10")][0], layer = "normalized")

# %%
plotdata = pd.DataFrame({
    "expression":sc.get.obs_df(transcriptome.adata, transcriptome.gene_id("SPI1"), layer = "normalized"),
    "phase":adata.obs["phase"],
})
plotdata.groupby("phase").mean()

# %%
# plotdata["expression"] = (plotdata["expression"] > 0).astype(int)

# %%
sns.boxplot(data = plotdata, x = "phase", y = "expression")

# %% [markdown]
# ### 100k

# %%
selected_transcripts = pickle.load((folder_data_preproc / "selected_transcripts.pkl").open("rb")).loc[transcriptome.var.index]
regions = chd.data.regions.Regions.from_transcripts(
    selected_transcripts, [-100000, 100000], dataset_folder / "regions" / "100k100k", overwrite = True
)

# %%
fragments_file = folder_data_preproc / "atac_fragments.tsv.gz"
fragments = chd.data.Fragments(dataset_folder / "fragments" / "100k100k")
fragments.regions = regions
fragments = chd.data.Fragments.from_fragments_tsv(
    fragments_file=fragments_file,
    regions=regions,
    obs=transcriptome.obs,
    path=fragments.path,
    batch_size=1e8,
    overwrite = True
)

# %%
fragments.create_regionxcell_indptr(overwrite = True)

# %%
