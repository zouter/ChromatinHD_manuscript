# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns
sns.set_style('ticks')
# %config InlineBackend.figure_format='retina'

import pickle

import scanpy as sc

import pathlib

import tqdm.auto as tqdm

import chromatinhd as chd
import tempfile
import requests

# %%
data_folder = chd.get_output() / "data" / "eqtl" / "onek1k"

# %% [markdown]
# ### Some sharing bonanza

# %%
target = transcriptome.path
target_output = chd.get_output()
source_output = "/home/wsaelens/NAS2/wsaelens/projects/chromatinhd/chromatinhd_manuscript/output"
source = source_output / target.relative_to(target_output)
if not target.exists():
    # !rm -d {target}
    target.symlink_to(source)

# %%
target_output = chd.get_output()
source_output = "/home/wsaelens/NAS2/wsaelens/projects/chromatinhd/chromatinhd_manuscript/output"
target_files = [
    data_folder / "variants_info.pkl",
    data_folder / "final.bcf.gz",
    data_folder / "final.bcf.gz.csi",
    data_folder / "donors_info.csv",
    data_folder / "genes.csv"
]
source_files = [source_output / file.relative_to(target_output) for file in target_files]

# %%
for source_file, target_file in zip(source_files, target_files):
    assert source_file.exists()
    if target_file.is_symlink():
        if target_file.resolve() == source_file:
            pass
    else:
        target_file.symlink_to(source_file)

# %% [markdown]
# ## Load data

# %% tags=[]
transcriptome = chd.data.transcriptome.transcriptome.ClusterTranscriptome(data_folder / "transcriptome")

# %% tags=[]
adata2 = transcriptome.adata

# %% tags=[]
variants_info = pd.read_pickle(data_folder / "variants_info.pkl")
variants_info["ix"] = np.arange(len(variants_info))

# %% tags=[]
donors_info = pd.read_csv(data_folder / "donors_info.csv", index_col = 0)

# %% tags=[]
genes = pd.read_csv(data_folder / "genes.csv", index_col = 0)

# %% tags=[]
cluster_info = pd.DataFrame({"cluster":transcriptome.obs["cluster"].unique().sort_values()}).set_index("cluster")
cluster_info["ix"] = np.arange(len(cluster_info))

# %% tags=[]
final_file = data_folder /"final.bcf.gz"

# %% tags=[]
import cyvcf2
vcf = cyvcf2.VCF(final_file)

# %% tags=[]
adata2.var["std"] = adata2.X.std(0)

# %% tags=[]
genes["std"] = adata2.var["std"]

# %% tags=[]
(genes
     # .query("biotype == 'lncRNA'")
     .sort_values("std", ascending = False)
     .head(20)
)

# %% [markdown]
# ## Modelling

# %%
# make sure the order of the samples is the same in expression and the vcf
# this is not checked afterwards
donors_info = donors_info.reset_index().set_index("old").loc[vcf.samples].reset_index().set_index("donor")

# %% tags=[]
import xarray as xr

# expression = pd.DataFrame(adata2.X, columns = adata2.var.index)
expression = pd.DataFrame(adata2.raw.X, columns = adata2.var.index)
expression.index = pd.MultiIndex.from_frame(adata2.obs[["cluster", "donor"]])
expression = xr.DataArray(expression).unstack()
expression.values[np.isnan(expression.values)] = 0.

# make sure the order of the samples is the same in expression and the vcf
# this is not checked afterwards
expression = expression.sel(donor = donors_info.reset_index().set_index("old").loc[vcf.samples]["donor"].values)


# %% tags=[]
def get_types(window):
    # f"{gene_oi.chr}:{window[0]}-{window[1]}"
    # get types
    types = []
    variant_ids = []
    for variant in vcf(window):
        types.append(variant.gt_types.copy()) #! Copy is important due to repurposing of gt_types by cyvcf2
        variant_ids.append(variant.ID)
    types = np.stack(types, -1)
    types[types == 2] = 1
    types[types == 3] = 2
    return types, variant_ids


# %% tags=[]
# gene_oi = genes.query("symbol == 'KCNAB2'").iloc[0]
# gene_oi = genes.query("symbol == 'GNB1-DT'").iloc[0]
# gene_oi = genes.query("symbol == 'IL1B'").iloc[0]
gene_oi = genes.query("symbol == 'CCL4'").iloc[0]
# gene_oi = genes.query("symbol == 'XBP1'").iloc[0]
# gene_oi = genes.query("symbol == 'CCR6'").iloc[0]
# gene_oi = genes.query("symbol == 'IL2RA'").iloc[0]
# gene_oi = genes.query("symbol == 'CD27'").iloc[0]
# gene_oi = genes.query("symbol == 'ITGA4'").iloc[0]
# gene_oi = genes.query("symbol == 'RGS1'").iloc[0]
# gene_oi = genes.query("symbol == 'CLEC2D'").iloc[0]

# %% tags=[]
expression_oi = expression.sel(gene = gene_oi.name).transpose("donor", "cluster")

# %% tags=[]
window_size = 10**6
window = (gene_oi["tss"] - window_size, gene_oi["tss"] + window_size)

# get types
window_str = f"{gene_oi.chr}:{window[0]}-{window[1]}"
genotype, variant_ids = get_types(window_str)
genotype = xr.DataArray(genotype - 1, coords = [donors_info.index, pd.Index(variant_ids, name = "variant")])
# genotype = genotype.sel(variant = ["chr2:203830251:C:T"])

# %% tags=[]
variants_info_oi = variants_info.loc[genotype.coords["variant"].to_pandas()]

# %% tags=[]
clusters_info = pd.DataFrame({"cluster":pd.Series(adata2.obs["cluster"].unique()).sort_values().dropna()}).set_index("cluster")
clusters_info["ix"] = np.arange(len(clusters_info))
cluster_to_cluster_ix = clusters_info["ix"].to_dict()

genes["ix"] = np.arange(len(genes))
gene_to_gene_ix = genes["ix"].to_dict()

donors_info = donors_info

# %%
# plotdata = pd.DataFrame({
#     # "expression":expression_oi.to_pandas().unstack()
#     # "expression":np.log1p(expression_oi.to_pandas().unstack())
#     "expression":np.log1p((expression_oi / lib) * 10**6).to_pandas().unstack()
# })
# plotdata["genotype"] = genotype.sel(variant = variant_id).to_pandas().loc[plotdata.index.get_level_values("donor")].values
# plotdata = plotdata.reset_index()

# sns.boxplot(data = plotdata, x = "cluster", y = "expression", hue = "genotype")

# %% [markdown]
# ### Start modeling

# %% tags=[]
import torch

# %%
from chromatinhd.models.eqtl.mapping.v1 import Model

# %% tags=[]
lib = (expression.sum("gene")).transpose("donor", "cluster").astype(np.float32)

# %% tags=[]
n_clusters = len(clusters_info)
n_donors = len(donors_info)
n_variants = len(variants_info_oi)

# %% tags=[]
genotype_torch = torch.from_numpy(genotype.values)
expression_oi_torch = torch.from_numpy(expression_oi.values)

# %%
baseline = (expression_oi / (lib + 1e-8)).mean("donor").values
baseline_torch = torch.from_numpy(baseline)

# %%
lib_torch = torch.from_numpy(lib.values)

# %%
model = Model(n_clusters, n_donors, n_variants, lib_torch, baseline_torch)

# %%
model = model.to("cuda")
genotype_torch = genotype_torch.to("cuda")
expression_oi_torch = expression_oi_torch.to("cuda")
model.lib = model.lib.to("cuda")

# %%
optim = torch.optim.Adam(model.parameters(), lr = 1e-2)

# %%
n_epochs = 500
checkpoint_every_epoch = 100

# %%
for epoch in range(n_epochs):
    elbo = model.forward(genotype_torch, expression_oi_torch)
    elbo.backward()
    optim.step()
    optim.zero_grad()
    if (epoch % checkpoint_every_epoch) == 0:
        print(elbo.item())

# %%
model2 = Model(n_clusters, n_donors, n_variants, lib_torch, baseline_torch, apply_genotype_effect=False)
model2.lib = model2.lib.to("cuda")

# %%
model2 = model2.to("cuda")
optim = torch.optim.Adam(model2.parameters(), lr = 1e-2)
for epoch in range(n_epochs):
    elbo = model2.forward(genotype_torch, expression_oi_torch)
    elbo.backward()
    optim.step()
    optim.zero_grad()
    if (epoch % checkpoint_every_epoch) == 0:
        print(elbo.item())

# %% [markdown]
# ### Interpret models

# %%
fc_log_mu = xr.DataArray(model.fc_log_mu.detach().cpu().numpy(), coords = [cluster_info.index, variants_info_oi.index])

# %%
scores = fc_log_mu.to_pandas().unstack().to_frame(name = "fc_log_mu")

# %%
scores.sort_values("fc_log_mu")

# %%
with torch.no_grad():
    elbo = model.forward(genotype_torch, expression_oi_torch)
    likelihood1 = model.get_likelihood().cpu().detach()
    elbo2 = model.forward(genotype_torch, expression_oi_torch)
    likelihood2 = model2.get_likelihood().cpu().detach()

# %%
lr = (likelihood1 - likelihood2).sum(0)

# %%
scores["lr"] = lr.T.flatten()

# %%
scores.groupby("cluster")["lr"].sum()

# %% [markdown]
# ### Interpret

# %%
# variant_id, cluster_id = scores.sort_values("fc_log_mu", ascending = False).index[0]
# variant_id, cluster_id = scores.query("cluster == 'cDCs'").sort_values("fc_log_mu", ascending = False).index[0]
variant_id, cluster_id = scores.query("cluster == 'CD4 T'").sort_values("lr", ascending = False).index[0]
# variant_id = (scores.xs("NK", level = "cluster")["lr"] - scores.xs("CD4 T", level = "cluster")["lr"]).sort_values(ascending = False).index[1]
# variant_id = variants_info.query("rsid == 'rs207253'").index[0]

# %%
scores.loc[variant_id].style.bar()

# %%
sns.heatmap(pd.DataFrame(np.corrcoef(scores["fc_log_mu"].unstack().T), index = cluster_info.index, columns = cluster_info.index))

# %%
plotdata = pd.DataFrame({
    # "expression":(expression_oi).to_pandas().unstack()
    "expression":np.log1p((expression_oi / lib) * 10**6).to_pandas().unstack()
})
plotdata["genotype"] = genotype.sel(variant = variant_id).to_pandas().loc[plotdata.index.get_level_values("donor")].values
plotdata = plotdata.reset_index()

sns.boxplot(data = plotdata, x = "cluster", y = "expression", hue = "genotype")

# %% [markdown]
# ### Visualize differential

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

dataset_name = "pbmc10k_leiden_0.1"
folder_data_preproc = folder_data / dataset_name

# %%
fragments = chd.data.fragments.ChunkedFragments(folder_data_preproc / "fragments")

# %%
chr = gene_oi["chr"]

# %%
chromosome_start = fragments.chromosomes.loc[chr]["position_start"]

# %% tags=[]
window_chunks = (
    (chromosome_start + window[0]) // fragments.chunk_size,
    (chromosome_start + window[1]) // fragments.chunk_size
)

# %%
chunks_from, chunks_to = fragments.chunkcoords_indptr[window_chunks[0]], fragments.chunkcoords_indptr[window_chunks[1]]

# %%
coordinates = (
    fragments.chunkcoords[chunks_from:chunks_to].to(torch.int64) * fragments.chunk_size
    + fragments.relcoords[chunks_from:chunks_to]
    - chromosome_start
)
clusters = fragments.clusters[chunks_from:chunks_to]

# %%
cluster_info = pickle.load((folder_data_preproc / ("cluster_info.pkl")).open("rb"))

# %%
# variants_query = variants_info_oi.query("rsid in ['rs207253', 'rs10944479']")
# variants_query = variants_info_oi.query("rsid in ['rs60849819']") # BACH2 rs60849819
variants_query = variants_info_oi.loc[[variant_id]]
variants_query = variants_info_oi.loc[variants_info_oi["rsid"].isin(qtl_mapped["rsid"])]
# variants_query = variants_info_oi.query("variant in ['chr2:203930034:T:C']")

# %%
window_oi = window
bins = np.linspace(*window_oi, int(np.diff(window_oi)//1000))

# window_oi = [variants_query.iloc[0]["pos"] - 50000, variants_query.iloc[0]["pos"] + 50000]
# bins = np.linspace(*window_oi, 500)

# %%
main = chd.grid.Grid(padding_height=0)
fig = chd.grid.Figure(main)

for cluster_ix, cluster in enumerate(cluster_info.index):
    # histogram
    main[cluster_ix * 2, 0] = panel = chd.grid.Panel((10, 0.5))
    ax = panel.ax
    ax.set_ylabel(cluster, rotation = 0, ha = "right", va = "center")
    ax.set_xticks([])
    
    coordinates_cluster = coordinates[clusters == cluster_ix]
    ax.hist(coordinates_cluster, bins = bins, lw = 0)
    ax.set_ylim(0, 0.03 * cluster_info.loc[cluster, "n_cells"])
    ax.axvline(gene_oi["tss"], dashes = (2, 2), color = "grey")
    ax.set_xlim(*window_oi)
    
    # lr
    main[cluster_ix * 2 + 1, 0] = panel = chd.grid.Panel((10, 0.5))
    ax = panel.ax
    
    plotdata_variants = variants_info_oi.copy()
    plotdata_variants["lr"] = scores.xs(cluster, level = "cluster")["lr"]
    
    ax.axvline(gene_oi["tss"], dashes = (2, 2), color = "grey")
    ax.scatter(plotdata_variants["pos"], plotdata_variants["lr"], s = 1, color = "red")
    ax.axhline(np.log(10), zorder = -10, lw = 1)
    # ax.set_ylim(np.log(10))
    
    for _, variant in variants_query.iterrows():
        ax.axvline(variant["pos"], zorder = -10, lw = 1)
    ax.set_xlim(*window_oi)
    
main[0, -1]
fig.plot()

# %% [markdown]
# ## LD

# %%
fig, ax = plt.subplots()
ld = pd.DataFrame(np.abs(np.corrcoef(genotype.T)), genotype.coords["variant"].to_pandas(), genotype.coords["variant"].to_pandas())
mappable = ax.matshow(ld, vmin = 0, vmax = 1)
fig.colorbar(mappable)
ax.axhline(variants_info_oi.index.to_list().index(variant_id))
ax.axvline(variants_info_oi.index.to_list().index(variant_id))

# %%
variants_info_oi.loc[variant_id]

# %%
variants_info_oi["ld"] = ld[variant_id]

# %%
fig, ax = plt.subplots()
ax.scatter(variants_info_oi["pos"], variants_info_oi["ld"])

# %% [markdown]
# ## Overlap with GWAS

# %%
motifscan_name = "gwas_immune"

# %%
folder_qtl = chd.get_output() / "data" / "qtl" / "hs" / "gwas"
folder_qtl.mkdir(exist_ok = True, parents=True)
qtl_mapped = pd.read_pickle(folder_qtl / ("qtl_mapped_" + motifscan_name + ".pkl"))

# %%
qtl_mapped["found"] = qtl_mapped["snp"].isin(variants_info["rsid"])

# %%
pd.Series(qtl_mapped["snp_main"].unique()).isin(variants_info["rsid"]).mean()

# %%
qtl_mapped["found"].mean()

# %% [markdown]
# ### Enrichment

# %%
scores["significant"] = scores["lr"] >= np.log(10)

# %%
variants_info_oi["significant_any"] = scores.groupby("variant")["significant"].any()

# %%
variants_info_oi["gwas"] = variants_info_oi["rsid"].isin(qtl_mapped["rsid"])

# %%
contingency = pd.crosstab(variants_info_oi["significant_any"], variants_info_oi["gwas"]).values

# %%
import fisher

# %%
fisher.pvalue(*contingency.flatten())

# %%
(contingency[0, 0] * contingency[1, 1])/(contingency[0, 1] * contingency[1, 0])

# %%
variants_info_oi["significant_any"].sum(), variants_info_oi["gwas"].sum()

# %%
