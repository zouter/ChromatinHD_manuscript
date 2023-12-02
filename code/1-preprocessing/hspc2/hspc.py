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
dataset_name = "hspc"

folder_data_preproc = folder_data / dataset_name / "MV2"
folder_data_preproc.mkdir(exist_ok = True, parents = True)
genome = "GRCh38"

# %%
folder_dataset = chd.get_output() / "datasets" / "hspc"

# %% [markdown]
# ## Create transcriptome

# %%
transcriptome = chd.data.Transcriptome(folder_dataset / "transcriptome")

# %%
adata = sc.read_10x_mtx(folder_data_preproc)

# %% [markdown]
# ### Read and process

# %%
adata.var.index.name = "symbol"
adata.var = adata.var.reset_index()
adata.var.index = adata.var["gene_ids"]
adata.var.index.name = "gene"

# %%
print(adata.obs.shape[0])
sc.pp.filter_cells(adata, min_counts=1000)
print(adata.obs.shape[0])
sc.pp.filter_cells(adata, min_genes=200)
print(adata.obs.shape[0])
sc.pp.filter_genes(adata, min_cells=100)
print(adata.var.shape[0])

# %%
transcripts = chd.biomart.get_transcripts(chd.biomart.Dataset.from_genome(genome), gene_ids=adata.var.index.unique())
pickle.dump(transcripts, (folder_data_preproc / 'transcripts.pkl').open("wb"))

# %%
# only retain genes that have at least one ensembl transcript
adata = adata[:, adata.var.index.isin(transcripts["ensembl_gene_id"])]

# %%
sc.external.pp.scrublet(adata)

# %%
adata.obs["doublet_score"].plot(kind="hist")

# %%
# adata.obs["doublet"] = (adata.obs["doublet_score"] > 0.1).astype("category")
adata.obs["doublet"] = (adata.obs["doublet_score"] > 0.5).astype("category")

print(adata.obs.shape[0])
adata = adata[~adata.obs["doublet"].astype(bool)]
print(adata.obs.shape[0])

# %%
size_factor = np.median(np.array(adata.X.sum(1)))
adata.uns["size_factor"] = size_factor

# %%
adata.raw = adata

# %%
sc.pp.normalize_total(adata, size_factor)

# %%
sc.pp.log1p(adata)

# %%
sc.pp.pca(adata)

# %%
sc.pp.highly_variable_genes(adata)

# %%
sc.pp.neighbors(adata)
sc.tl.umap(adata)

# %%
sc.pl.umap(adata)

# %%
import pickle

# %%
adata.layers["normalized"] = adata.X
adata.layers["counts"] = adata.raw.X

# %%
import magic

magic_operator = magic.MAGIC(knn=30, solver = "approximate")
X_smoothened = magic_operator.fit_transform(adata.X)
adata.layers["magic"] = X_smoothened

# %%
pickle.dump(adata, (folder_data_preproc / 'adata.pkl').open("wb"))

# %% [markdown]
# ## Interpret and subset

# %%
adata = pickle.load((folder_data_preproc / 'adata.pkl').open("rb"))


# %%
def get_gene_id(symbols):
    return adata.var.reset_index().set_index("symbol").loc[symbols, "gene"].values


# %%
sc.tl.leiden(adata, resolution=2.0)
sc.pl.umap(adata, color="leiden")

# %%
sc.pl.umap(adata, color = "n_counts")

# %%
adata.var.loc[adata.var.symbol.str.startswith("MT")].sort_values("dispersions_norm")

# %%
adata.var.sort_values("dispersions_norm", ascending = False).query("means > 0.1").head(20)

# %%
# !wget https://raw.githubusercontent.com/scverse/scanpy_usage/master/180209_cell_cycle/data/regev_lab_cell_cycle_genes.txt

# %%
cell_cycle_genes = [x.strip() for x in open('./regev_lab_cell_cycle_genes.txt')]
s_genes = cell_cycle_genes[:43]
s_genes = get_gene_id([g for g in s_genes if g in adata.var["symbol"].tolist()])
g2m_genes = cell_cycle_genes[43:]
g2m_genes = get_gene_id([g for g in g2m_genes if g in adata.var["symbol"].tolist()])

sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes, use_raw = False)

# %%
fig, ax = plt.subplots()
phase_colors = pd.Series({'G1':'tab:blue', 'G2M':'tab:orange', 'S':'tab:green'})
ax.scatter(adata.obs["S_score"], adata.obs["G2M_score"], s=1, c = phase_colors[adata.obs["phase"]])

# %%
sc.tl.rank_genes_groups(adata, "leiden", use_raw = False)

# %%
sc.pl.umap(adata, color = ["leiden", "phase"], legend_loc = "on data", title = "leiden")

# %%
diffexp = sc.get.rank_genes_groups_df(adata, group='3')
diffexp["symbol"] = adata.var.loc[diffexp["names"], "symbol"].values
diffexp.sort_values("logfoldchanges", ascending = False).head(20)

# %%
topscores

# %%
topscores = diffexp.set_index(["symbol", "group"])["logfoldchanges"].unstack()
top = topscores.idxmax(1)
topscores.loc[top == "1"]["1"].sort_values(ascending = False).head(20)

# %%
diffexp = sc.get.rank_genes_groups_df(adata, group = None)
diffexp["symbol"] = adata.var.loc[diffexp["names"], "symbol"].values
diffexp.query("symbol == 'AZU1'").sort_values("scores", ascending = False)

# %%
sc.pl.umap(adata, color = ["S_score", "G2M_score", "phase"], vmin = 0)
symbols = ["PCNA", "MKI67", "MCM3"]
sc.pl.umap(adata, color = get_gene_id(symbols), vmin = 0, title = symbols)

# %%
cycleclusters = adata.obs.groupby(["phase", "leiden"]).size().unstack()
cycleclusters = cycleclusters.divide(cycleclusters.sum(1), 0)
cycleclusters.loc["G1"].sort_values(ascending = False).head(20)

# %%
with plt.rc_context({"figure.figsize": (1.5, 1.5)}):
    # 0
    symbols = ["CD74", "MSI2", "CD34"]
    sc.pl.umap(adata, color = get_gene_id(symbols), layer = "normalized", title = symbols)

    # 1
    symbols = ["MPO", "STOX2", "AZU1"]
    sc.pl.umap(adata, color = get_gene_id(symbols), layer = "normalized", title = symbols)

    # 4
    symbols = ["CTSG", "FBLN5"]
    sc.pl.umap(adata, color = get_gene_id(symbols), layer = "normalized", title = symbols)

    # 5
    symbols = ["SOX4", "IKZF2"]
    sc.pl.umap(adata, color = get_gene_id(symbols), layer = "normalized", title = symbols)

    # 3
    symbols = ["NPPA"]
    sc.pl.umap(adata, color = get_gene_id(symbols), layer = "normalized", title = symbols)


    # 20 # progenitor of 22?
    symbols = ["AFF3"]
    sc.pl.umap(adata, color = get_gene_id(symbols), layer = "normalized", title = symbols)

    # 22
    symbols = ["TGFBI", "CD74", "AFF3"]
    sc.pl.umap(adata, color = get_gene_id(symbols), layer = "normalized", title = symbols)

    # 10
    symbols = ["HBB"]
    sc.pl.umap(adata, color = get_gene_id(symbols), layer = "normalized", title = symbols)

    # 13
    symbols = ["ITGA2B", "THBS1", "GATA1", "PF4", "VWF"]
    sc.pl.umap(adata, color = get_gene_id(symbols), layer = "normalized", title = symbols)
    
    # 21
    symbols = ["KIT", "CD44"]
    sc.pl.umap(adata, color = get_gene_id(symbols), layer = "normalized", title = symbols)
    
    # 23 # progenitor of 21?
    symbols = ["IL18R1", "IL5RA", "HDC"]
    sc.pl.umap(adata, color = get_gene_id(symbols), layer = "normalized", title = symbols)

    # 15
    symbols = ["RPS18"]
    sc.pl.umap(adata, color = get_gene_id(symbols), layer = "normalized", title = symbols)

    # 19
    symbols = ["RUNX1", "GATA2"]
    sc.pl.umap(adata, color = get_gene_id(symbols), layer = "normalized", title = symbols)

    # 27
    symbols = ["TCF4", "AFF3"]
    sc.pl.umap(adata, color = get_gene_id(symbols), layer = "normalized", title = symbols)

# %%
symbols = ["MPO", "CD34", "PF4", "HBB", "FOS", "RGS6", "VCAN", "VWF", "CD74", "CNTNAP2", "QKI", "GATA1", "THBS1", "PDE4D", "LYZ", "CLC", , , "", ]
sc.pl.umap(adata, color = get_gene_id(symbols), layer = "normalized", title = symbols)

# %%
adatanocc = adata[(adata.obs["phase"] == "G1") & (adata.obs["doublet_score"] < 0.1)].copy()
sc.pp.neighbors(adatanocc)
sc.tl.umap(adatanocc)
sc.tl.leiden(adatanocc, resolution=1., key_added="leiden_nocc")

# %%
sc.pl.umap(adatanocc, color = ["n_counts", "leiden", "leiden_nocc"])

# %%
symbols = ["CD34", "AFF3", "MPO", "HBB", "VWF", "TAL1", "IKZF2", "RUNX1", "STOX2", "AZU1"]
sc.pl.umap(adatanocc, color = get_gene_id(symbols), layer = "magic", title = symbols)

# %%
sc.pl.umap(adatanocc, color = ["leiden_nocc", "leiden"], legend_loc = "on data")

# %%
pickle.dump(adata, (folder_data_preproc / 'adata_annotated.pkl').open("wb"))

# %% [markdown]
# ## GMP

# %%
adata_gmp = adatanocc[adatanocc.obs["leiden_nocc"].isin(["0", "9", "2", "5", "3"]) & ~adatanocc.obs["leiden"].isin(["5"])].copy()
pickle.dump(adata_gmp, (folder_data_preproc / 'adata_gmp.pkl').open("wb"))

# %%
sc.pl.umap(adata_gmp)

# %% [markdown]
# ## TSS

# %%
adata = pickle.load((folder_data_preproc / 'adata_annotated.pkl').open("rb"))

# %%
transcripts = pickle.load((folder_data_preproc / 'transcripts.pkl').open("rb"))
transcripts = transcripts.loc[transcripts["ensembl_gene_id"].isin(adata.var.index)]

# %%
fragments_file = folder_data_preproc / "atac_fragments.tsv.gz"
selected_transcripts = chd.data.regions.select_tss_from_fragments(transcripts, fragments_file)

# %%
pickle.dump(selected_transcripts, (folder_data_preproc / 'selected_transcripts.pkl').open("wb"))

# %%
adata.var["means"].shape

# %%
np.log(transcripts.groupby("ensembl_gene_id")["n_fragments"].max()).shape

# %%
plt.scatter(adata.var["means"], np.log(transcripts.groupby("ensembl_gene_id")["n_fragments"].max())[adata.var.index])

# %%
pickle.dump(selected_transcripts, (folder_data_preproc / 'selected_transcripts.pkl').open("wb"))

# %% [markdown]
# ## Preprocess

# %%
dataset_folder = chd.get_output() / "datasets" / "hspc"
dataset_folder.mkdir(exist_ok=True, parents=True)

# %%
adata = pickle.load((folder_data_preproc / "adata_annotated.pkl").open("rb"))

# %% [markdown]
# ### Create transcriptome

# %%
# from fulco 2019 dataset, some of these may not be differential but we still want to keep them
desired_genes = ['GATA1', 'PQBP1', 'HDAC6', 'NUCB1', 'FTL', 'BCAT2', 'PRDX2', 'PLP2',
       'RNASEH2A', 'DNASE2', 'KLF1', 'WDR83OS', 'LYL1', 'RPN1',
       'SEC61A1', 'CNBP', 'NFE2', 'ITGA5', 'HNRNPA1', 'COPZ1', 'BAX',
       'PPP1R15A', 'FUT1', 'RAD23A', 'CALR', 'DHPS', 'JUNB', 'H1FX', 'RAB7A']

# %%
genes_oi_1 = adata.var.sort_values("dispersions_norm").tail(5000).index
genes_oi_2 = adata.var.reset_index().set_index("symbol").loc[desired_genes, "gene"].values

genes_oi = np.concatenate([genes_oi_1, genes_oi_2[~np.isin(genes_oi_2, genes_oi_1)]])
transcriptome = chd.data.transcriptome.Transcriptome.from_adata(adata[:, genes_oi], path=dataset_folder / "transcriptome")

# %%
sc.pl.umap(adata, color = adata.var.index[(adata.var["symbol"] == "ITGA5")][0], layer = "magic")

# %% [markdown]
# ### 10k

# %%
selected_transcripts = pickle.load((folder_data_preproc / "selected_transcripts.pkl").open("rb")).loc[transcriptome.var.index]
regions = chd.data.regions.Regions.from_transcripts(
    selected_transcripts, [-10000, 10000], dataset_folder / "regions" / "10k10k"
)

# %%
fragments_file = folder_data_preproc / "atac_fragments.tsv.gz"
fragments = chd.data.Fragments(dataset_folder / "fragments" / "10k10k")
fragments.regions = regions
fragments = chd.data.Fragments.from_fragments_tsv(
    fragments_file=fragments_file,
    regions=regions,
    obs=transcriptome.obs,
    path=fragments.path,
    overwrite = True,
)

# %%
fragments.create_regionxcell_indptr()

# %% [markdown]
# ### 100k

# %%
selected_transcripts = pickle.load((folder_data_preproc / "selected_transcripts.pkl").open("rb")).loc[transcriptome.var.index]
regions = chd.data.regions.Regions.from_transcripts(
    selected_transcripts, [-100000, 100000], dataset_folder / "regions" / "100k100k"
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
    batch_size=1e8
)

# %%
fragments.create_regionxcell_indptr(overwrite = True)

# %% [markdown]
# ### 500k

# %%
dataset_name = "hspc_focus"
dataset_folder = chd.get_output() / "datasets" / dataset_name

# %%
folder_data_preproc2 = chd.get_output() / "data" / dataset_name
folder_data_preproc2.mkdir(exist_ok = True, parents = True)
# !ln -s {folder_data_preproc}/atac_fragments.tsv.gz {folder_data_preproc2}/atac_fragments.tsv.gz
# !ln -s {folder_data_preproc}/atac_fragments.tsv.gz.tbi {folder_data_preproc2}/atac_fragments.tsv.gz.tbi
# !ln -s {chd.get_output()}/peaks/hspc/macs2_leiden_0.1_merged {chd.get_output()}/peaks/{dataset_name}/macs2_leiden_0.1_merged
# !ln -s {chd.get_output()}/peaks/hspc/cellranger {chd.get_output()}/peaks/{dataset_name}/cellranger

# %%
desired_genes = ["GATA1", "KLF1", "CALR"]
genes_oi = adata.var.reset_index().set_index("symbol").loc[desired_genes, "gene"].values
transcriptome = chd.data.transcriptome.Transcriptome.from_adata(adata[:, genes_oi], path=dataset_folder / "transcriptome")

# %%
selected_transcripts = pickle.load((folder_data_preproc / "selected_transcripts.pkl").open("rb")).loc[transcriptome.adata.var.index]
regions = chd.data.regions.Regions.from_transcripts(
    selected_transcripts, [-500000, 500000], dataset_folder / "regions" / "500k500k"
)

# %%
fragments_file = folder_data_preproc / "atac_fragments.tsv.gz"
fragments = chd.data.Fragments(dataset_folder / "fragments" / "500k500k")
fragments.regions = regions
fragments = chd.data.Fragments.from_fragments_tsv(
    fragments_file=fragments_file,
    regions=regions,
    obs=transcriptome.obs,
    path=fragments.path,
)

# %%
fragments.create_regionxcell_indptr()
