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
sc.tl.leiden(adata, resolution=2.0)
sc.pl.umap(adata, color="leiden")


# %%
def get_gene_id(symbols):
    return adata.var.reset_index().set_index("symbol").loc[symbols, "gene"].values


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
transcriptome.var.sort_values("dispersions_norm").iloc[:500]

# %%
sc.pl.umap(adata, color=[*transcriptome.gene_id(["NOSIP", "IKZF1"])], use_raw = False, layer = "normalized")

# %%
sc.pl.umap(adata, color=["phase", "S_score", "G2M_score", *transcriptome.gene_id(["GATA1", "FUT1", "CDK1", "MKI67", "PCNA"])], use_raw = False, layer = "magic")

# %%
adata2 = adata[adata.obs["phase"] == "G1"]
sc.pp.pca(adata2)
sc.pp.neighbors(adata2)
sc.tl.umap(adata2)
sc.tl.leiden(adata2, resolution=2.0)
sc.pl.umap(adata2, color="leiden")

# %%
sc.tl.rank_genes_groups(adata2, "leiden", use_raw = False)

# %%
sc.pl.umap(adata2, color = ["leiden", "phase"], legend_loc = "on data", title = "leiden")

# %%
diffexp = sc.get.rank_genes_groups_df(adata2, group='21')
diffexp["symbol"] = adata2.var.loc[diffexp["names"], "symbol"].values
diffexp.sort_values("logfoldchanges", ascending = False).head(20)
# print("\n".join(diffexp.sort_values("logfoldchanges", ascending = False).head(60)["symbol"].str.capitalize().tolist()))7

# %%
symbols_oi = ["CD274"]
sc.pl.umap(adata2, color = get_gene_id(symbols_oi), vmin = 0, title = symbols_oi)

# %%
diffexp = sc.get.rank_genes_groups_df(adata2, group = None)
diffexp["symbol"] = adata.var.loc[diffexp["names"], "symbol"].values
diffexp.query("symbol == 'TUBG2'").sort_values("scores", ascending = False)

# %%
topscores[["19", "3", "5"]].mean(1).sort_values()

# %%
topscores = diffexp.set_index(["symbol", "group"])["logfoldchanges"].unstack()
top = topscores.idxmax(1)
topscores.loc[top == "3"]["3"].sort_values(ascending = False).head(20)

# %%
sc.pl.umap(adata, color = ["S_score", "G2M_score", "phase"], vmin = 0)
symbols = ["PCNA", "MKI67", "MCM3"]
sc.pl.umap(adata, color = get_gene_id(symbols), vmin = 0, title = symbols)

# %%
cycleclusters = adata.obs.groupby(["phase", "leiden"]).size().unstack()
cycleclusters = cycleclusters.divide(cycleclusters.sum(1), 0)
cycleclusters.loc["G1"].sort_values(ascending = False).head(20)

# %%
symbols_oi = ["GATA1"]
sc.pl.umap(adata2, color = get_gene_id(symbols_oi), vmin = 0, title = symbols_oi)

# %%
import io
marker_annotation = pd.DataFrame([
    ["gmp",             ["MPO", "STOX2", "AZU1", "LMO4", "ELANE"]],
    ["granulocyte",     ["DIP2C", "ITK"]],
    ["hspc",            ["HLF"]],
    ["mpp",            ["STAB1", "CD52", "SPINK2", "SERPINI2", "UMODL1"]],
    ["lmpp",            ["ILDR2", "MPO"]],
    ["erythro",             ["HBB", "KLF1"]],
    ["prog DC",         ["LYZ"]],
    ["megakaryocyte",        ["PF4", "GP5", "COL6A3"]],
    ["megakaryocyte progenitors",      ["IGSF3"]],
    # ["neutrophil???",      ["NPPA", "RIBC1", "EXD2", "LGR6", "TEX15"]],
    ["unknown 1",      ["HIST2H3D"]],
    ["mep",      ["IKZF2", "PURG"]],
    ["macrophage precursor",      ["CSF1R"]],
    ["lymphoid/ilc?",      ["IL5RA"]],
    ["pro-b",      ["JCHAIN", "SPIB"]],
    ["unknown",      ["GPR3", "WNT11"]],
    # ["mep",      ["AGTR1", "ST8SIA1", "MTUS1", "AQP3"]],
], columns = ["celltype", "symbols"]).set_index("celltype")

# %%
import chromatinhd.utils.scanpy
cluster_celltypes = chd.utils.scanpy.evaluate_partition(
    adata, marker_annotation["symbols"].to_dict(), "symbol", partition_key="leiden"
).idxmax()

adata.obs["celltype"] = adata.obs["celltype"] = cluster_celltypes[
    adata.obs["leiden"]
].values
adata.obs["celltype"] = adata.obs["celltype"] = adata.obs[
    "celltype"
].astype(str)



import chromatinhd.utils.scanpy
cluster_celltypes = chd.utils.scanpy.evaluate_partition(
    adata2, marker_annotation["symbols"].to_dict(), "symbol", partition_key="leiden"
).idxmax()

adata2.obs["celltype"] = adata2.obs["celltype"] = cluster_celltypes[
    adata2.obs["leiden"]
].values
adata2.obs["celltype"] = adata2.obs["celltype"] = adata2.obs[
    "celltype"
].astype(str)

# %%
sc.pl.umap(adata2, color = ["celltype", "leiden", "phase"], legend_loc = "on data")

# %%
sc.pl.umap(adata, color = ["celltype", "leiden", "phase"], legend_loc = "on data")

# %%
symbols_oi = ["KLF1"]
sc.pl.umap(adata, color = get_gene_id(symbols_oi), vmin = 0, title = symbols_oi)

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
# ## CC regress

# %%
adata3 = sc.pp.regress_out(adata, ["S_score", "G2M_score"], copy=True)
sc.pp.pca(adata3)
sc.pp.neighbors(adata3)
sc.tl.umap(adata3)
sc.tl.leiden(adata3, resolution=2.0)
sc.pl.umap(adata3, color=["leiden", "phase"])

# %%
import io
marker_annotation = pd.DataFrame([
    ["GMP",             ["MPO", "STOX2", "AZU1", "LMO4", "ELANE"]],
    ["Granulocyte 1",     ["DIP2C", "ITK"]],
    ["HSPC",            ["HLF"]],
    ["MPP",            ["STAB1", "CD52", "SPINK2", "SERPINI2", "UMODL1"]],
    ["LMPP",            ["ILDR2", "MPO"]],
    ["Erythroblast",             ["HBB", "KLF1"]],
    ["Myeloid",         ["LYZ"]],
    ["Megakaryocyte",        ["PF4", "GP5", "COL6A3"]],
    ["Megakaryocyte progenitors",      ["IGSF3"]],
    ["Unknown 1",      ["HIST2H3D"]],
    ["MEP",      ["IKZF2", "PURG"]],
    ["Granulocyte 2",      ["IL5RA"]],
    ["Myeloblast",      ["JCHAIN", "SPIB"]],
    ["unknown",      ["GPR3", "WNT11"]],
], columns = ["celltype", "symbols"]).set_index("celltype")

# %%
import chromatinhd.utils.scanpy
cluster_celltypes = chd.utils.scanpy.evaluate_partition(
    adata3, marker_annotation["symbols"].to_dict(), "symbol", partition_key="leiden"
).idxmax()

adata3.obs["celltype"] = adata3.obs["celltype"] = cluster_celltypes[
    adata3.obs["leiden"]
].values
adata3.obs["celltype"] = adata3.obs["celltype"] = adata3.obs[
    "celltype"
].astype(str)

# %%
sc.pl.umap(adata3, color = ["celltype", "leiden", "phase"], legend_loc = "on data")

# %%
fixes = {
    "19":"Megakaryocyte progenitors",
    "12":"Megakaryocyte progenitors",
    "22":"Erythrocyte precursors",
    "6":"Erythrocyte precursors",
    "4":"Megakaryocyte-erythrocyte gradient",
    "3":"MEP",
    "23":"B-cell precursors",
    "16":"Granulocyte precursor",
    "14":"Unknown 2",
}
adata3.obs["celltype"] = adata3.obs["celltype"].astype(str)
for k, v in fixes.items():
    adata3.obs.loc[adata3.obs["leiden"] == k, "celltype"] = v

# %%
sc.pl.umap(adata3, color = ["celltype", "leiden", "phase", "n_counts"], legend_loc = "on data")

# %%
symbols_oi = ["CD74", "JCHAIN", "SPIB", "CD63", "IGLL1", "IL5RA", "WNT11"]
sc.pl.umap(adata3, color = get_gene_id(symbols_oi), vmin = 0, title = symbols_oi)

# %% [markdown]
# ##### Look at diffexp

# %%
sc.tl.rank_genes_groups(adata3, "leiden")

# %%
diffexp = sc.get.rank_genes_groups_df(adata3, group='23')
diffexp["symbol"] = adata3.var.loc[diffexp["names"], "symbol"].values
diffexp.sort_values("logfoldchanges", ascending = False).head(20)

# %%
pickle.dump(adata3, (folder_data_preproc / 'adata_annotated.pkl').open("wb"))

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
folder_data_preproc

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
transcriptome = chd.data.transcriptome.Transcriptome.from_adata(adata[:, genes_oi], path=dataset_folder / "transcriptome", overwrite=True)

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
desired_genes = ["GATA1", "KLF1", "CALR", "H1FX"]
genes_oi = adata.var.reset_index().set_index("symbol").loc[desired_genes, "gene"].values
transcriptome = chd.data.transcriptome.Transcriptome.from_adata(adata[:, genes_oi], path=dataset_folder / "transcriptome")

# %%
genes_oi

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
    overwrite = True
)

# %%
fragments.create_regionxcell_indptr(overwrite = True)

# %% [markdown]
# ## Explore

# %%
dataset_folder = chd.get_output() / "datasets" / "hspc"
transcriptome = chd.data.transcriptome.Transcriptome(path=dataset_folder / "transcriptome")

# %%
import scanpy as sc

# %%
adata = pickle.load((folder_data_preproc / 'adata.pkl').open("rb"))[transcriptome.adata.obs.index]
adata.obs = transcriptome.obs
adata.obsm["X_umap2"] = transcriptome.adata.obsm["X_umap"]

# %%
sc.pl.embedding(adata, color = ["celltype", "phase"], basis = "umap2", legend_loc = "on data")

# %%
# inside = adata.obs["celltype"].isin(["HSPC", "LMPP", "MPP", "GMP"])
inside = adata.obs["celltype"].isin(["HSPC"])


# %% [markdown]
# ### Find path

# %%
def gene_id(symbols):
    return adata.var.reset_index().set_index("symbol").loc[symbols, "gene"].values


# %%
plotdata = transcriptome.adata.obs.copy()
plotdata["expression"] = sc.get.obs_df(adata, gene_id(["SPI1"]), layer = "normalized")

# %%
# fig, ax = plt.subplots()
# ax.scatter(plotdata["S_score"], plotdata["G2M_score"], c = plotdata["expression"], s= 1)

# %%
# sc.pl.embedding(adata, color = gene_id(["PCNA", "CDK1", "BIRC5", "HBB", "SPI1", "PTPRC"]), use_raw = False, layer = "normalized", basis = "umap2") 

# %%
plotdata["umap1"] = np.array(adata.obsm["X_umap2"][:, 0])
plotdata["umap2"] = np.array(adata.obsm["X_umap2"][:, 1])

# %%
plotdata = plotdata.iloc[:4999]

# %%
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd

# Example data
# df = pd.DataFrame({
#     'x': np.random.randint(0, 100, 100),
#     'y': np.random.randint(0, 100, 100),
#     'customdata': np.random.choice(['A', 'B', 'C'], 100),
# })

# Create a scatter plot
fig = go.Figure(data=go.Scatter(x=plotdata['umap1'], y=plotdata['umap2'], mode='markers', customdata=plotdata['expression']))
fig.update_layout(
    margin=dict(l=20, r=20, t=20, b=20), width = 500, height = 500, plot_bgcolor = "white")

# Initialize the Dash app
app = dash.Dash("test")

# Layout of the app
app.layout = html.Div([
    dcc.Graph(id='scatter-plot', figure=fig),
    html.Pre(id='selected-data', style={"whiteSpace": "pre-line"}),
])

# Callback to update the selection
@app.callback(
    Output('selected-data', 'children'),
    [Input('scatter-plot', 'selectedData')]
)
def display_selected_data(selectedData):
    print(selectedData)
    if selectedData is None:
        return "No data selected"
    return str(selectedData["lassoPoints"]) 
    # return str(selectedData)

app.run_server(debug = True, port = 8049)

# %%
# early myeloid
paths = {'x': [-3.141114001152576, -3.509996976452074, -3.632957968218573, -3.5509839737075737, -2.649270034086579, -2.157426067020582, -1.1737381328885879, -0.7638681603335904, -0.6818941658225909, -1.0507771411220888, -1.501634110932586], 'y': [0.337713352427159, -0.42216330326322976, -0.42216330326322976, -0.7116401244786159, -1.507701382820928, -2.1590242305555467, -2.30376264116324, -2.2313934358593936, -2.0866550252517, -1.1096707536497719, -0.27742489265553666]}

# all myeloid
# paths = {'x': [-3.309110435588785, -4.410358976924616, -4.451145959937054, -4.53271992596193, -4.043276129812672, -2.6565187073897745, -1.7592050811161346, -1.0250393868922474, -0.8211044718300565, -0.6579565397803039, -0.8618914548424947, -1.3105482679793146, -1.7999920641285727, -1.9223530131658872], 'y': [0.49249244532192643, -0.7042969389253823, -1.5746892183779704, -1.8648199781954997, -2.517614187784941, -2.6264132227165145, -2.4450814978305586, -2.118684393035838, -1.9010863231726909, -0.9218950087885293, -0.2691007991990882, 0.20236168550439704, 0.41995975536754404, 0.5650251352763087]}

# paths = {'x': [8.029670841869027, 7.907309892831712, 5.827173759197366, 6.194256606309309, 6.561339453421253, 6.642913419446129, 7.29550514764514, 8.355966705968532, 10.069019992490936, 11.53735138093871, 12.597812939262102, 13.128043718423799, 13.291191650473552, 13.413552599510867, 13.495126565535742], 'y': [2.617343319498325, 2.617343319498325, 1.387066829332934, 0.8804823922060081, 0.337713352427159, -0.13268648204784356, -0.7478247271305393, -1.1820399589536186, -1.4353321775170815, -1.3629629722132348, -0.7116401244786159, -0.27742489265553666, 0.08442113386369608, 2.2554972929790926, 2.5087895115425556]}
# paths = {'x': [-1.7737580021613741, -2.275711448823683, -2.538639444694416, -2.6581521700902035, -2.6581521700902035, -2.2996139939028404, -1.8693681824780044, -1.415219825974011, -0.12448239169950294, 0.01893287877544239, 0.18625069432954527, 0.9033270467042719, 1.333572858129108, 2.074551755582992, 2.337479751453725, 2.839433198116034, 3.2218739193825545, 3.795535001282336, 5.3252978863484195, 5.468713156823365, 5.588225882219152, 5.731641152694098, 5.731641152694098, 4.966759710161056, 3.9628528168364388, 3.150166284145082, 2.624310292403616, 1.9311364851080468, 1.4052804933665806, 1.21406013273332, 1.0945474073375323], 'y': [8.410500041529634, 7.8827167317083955, 7.291599424708609, 6.721593450101673, 6.130476143101887, 5.138243520637959, 4.462680884066774, 3.871563577066988, 1.5915396786392406, 1.2326470279607988, 0.5570843913896144, -1.3640468563596913, -2.0607208253237252, -2.6096154675378127, -2.7573947942877592, -2.9262854534305553, -2.9685081182162545, -2.905174121037706, -2.4829474731807157, -2.377390811216468, -2.166277487287973, -0.815152214145604, -0.49848222825286126, 1.4648716842821436, 3.449336929209998, 4.948241529102313, 6.130476143101887, 7.24937675992291, 8.55827936827958, 9.212730672457916, 9.466066661172109]}

# gmp
# paths = {'x': [-2.20400381358621, -2.8971776208817794, -3.7576692437314514, -3.8054743338897663, -3.446936157702403, -2.9927878011984097, -1.463024916132326, -1.080584194865805, 1.11844995241669, 1.9550390301872043, 2.3852848416120405, 3.0067510136701365, 3.197971374303397, 3.197971374303397, 2.1701619358996225, 1.6204033990789986, 1.1423524974958474, 1.0945474073375323], 'y': [8.051607390851192, 6.679370785315974, 5.264911514995056, 4.65268287560242, 4.188233562959731, 3.744895582709891, 2.9426649517816097, 2.8159969574245123, 2.879330954603061, 3.111555610924406, 3.3648915996385997, 4.20934489535258, 4.779350869959517, 5.602692833280648, 7.7138260725656, 8.642724697850978, 9.276064669636463, 9.276064669636463]}

polygon = np.array([paths["x"], paths["y"]]).T

import matplotlib.path as mpath
path = mpath.Path(polygon)
inside = path.contains_points(np.array([adata.obsm["X_umap2"][:, 0], adata.obsm["X_umap2"][:, 1]]).T)

# %%
adata.obs["inside"] = pd.Series(inside, index = adata.obs.index).astype(int)
sc.pl.embedding(adata, color = "inside", basis = "umap2")

# %% [markdown]
# ### Filter on inside

# %%
adata2 = adata[inside]
inside.sum()

# %%
sc.pl.embedding(adata2, color = gene_id(["SPI1", "CDK1", "PCNA", "HBB", "PTPRC"]), use_raw = False, layer = "normalized", basis = "umap2")

# %%
adata.obs["inside"] = pd.Categorical(inside)
sc.tl.rank_genes_groups(adata, groupby = "inside")
diffexp = sc.get.rank_genes_groups_df(adata, group = "True")
diffexp["symbol"] = adata.var.loc[diffexp["names"], "symbol"].values

# %%
plotdata2 = adata2.obs.copy()
plotdata2["umap1"] = adata2.obsm["X_umap2"][:, 0]
plotdata2["umap2"] = adata2.obsm["X_umap2"][:, 1]

# symbol = "BIRC5"
# symbol = "PCNA"
# symbol = "PTPRC"
# symbol = "SPI1"
# symbol = "GATA2"
# symbol = "GATA1"
# symbol = "FOS"
# symbol = "SPI1"
# symbol = "CCR2"
# symbol = "AFF3"
# symbol = "MPO"
# symbol = "ACTB"
# symbol = "E2F2"
# symbol = "MKI67"
symbol = "ERG"

plotdata2["expression"] = sc.get.obs_df(adata2, gene_id([symbol]), layer = "normalized")

# %%
plotdata2.groupby("phase")["expression"].mean()

# %%
sns.boxplot(x = "phase", y = "expression", data = plotdata2)

# %%
fig, ax = plt.subplots()
ax.scatter(plotdata2["G2M_score"], plotdata2["expression"])
sns.regplot(x = plotdata2["G2M_score"], y = plotdata2["expression"], ax = ax)

import scipy.stats
# (scipy.stats.linregress(plotdata2["G2M_score"], plotdata2["expression"]).slope)
np.exp(scipy.stats.linregress(plotdata2["G2M_score"], plotdata2["expression"]).slope)

# %%
sc.pl.umap(transcriptome.adata, color = [transcriptome.gene_id("CCR2"), transcriptome.gene_id("AFF3")], use_raw = False, layer = "normalized") 
