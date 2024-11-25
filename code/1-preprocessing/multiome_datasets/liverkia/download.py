# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: chromatinhd
#     language: python
#     name: python3
# ---

# %%
import polyptich as pp
pp.setup_ipython()

from matplotlib import legend
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")

import pickle

import scanpy as sc

import tqdm.auto as tqdm
import io

import chromatinhd as chd

import magic
import eyck
import scipy.sparse

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

dataset_name = "liverkia"
genome = "mm10"
organism = "mm"

folder_data_preproc = folder_data / dataset_name
folder_data_preproc.mkdir(exist_ok=True, parents=True)
folder_data_preproc = folder_data_preproc / "liver_control_JVG28"

folder_dataset = chd.get_output() / "datasets" / dataset_name

# %% [markdown]
# ## Download

# %%
from webdav3.client import Client

password = input("Password: ")
options = {
    "webdav_hostname": "https://cloud.irc.ugent.be/public/remote.php/webdav",
    "webdav_login": "wouters",
    "webdav_password": password,
}

client = Client(options)
client.download_sync(
    remote_path="IRC FileServer #04 [FS4] - (M-volume) Instrumentation {PUBLIC share}/u_mgu/multiome_kia-jo/liver_multiome_MusratiEtAl/",
    local_path=folder_data_preproc.parent,
)

# %%
import pysam

pysam.tabix_index(
    str(folder_data_preproc / "atac_fragments.tsv.gz"),
    seq_col=0,
    start_col=1,
    end_col=2,
)

# %% [markdown]
# ## Create transcriptome

# %%
transcriptome = eyck.modalities.Transcriptome(folder_dataset / "transcriptome")

# %%
adata = sc.read_10x_h5(folder_data_preproc / "liver_naive_JVG28_feature_bc_matrix.h5")
adata.var["symbol"] = adata.var.index
adata.var["gene"] = adata.var["gene_ids"]
adata.var = adata.var.set_index("gene")

# %%
transcripts = chd.biomart.get_transcripts(
    chd.biomart.Dataset.from_genome(genome), gene_ids=adata.var.index.unique()
)
pickle.dump(transcripts, (folder_data_preproc / "transcripts.pkl").open("wb"))

# %%
# only retain genes that have at least one ensembl transcript
adata = adata[:, adata.var.index.isin(transcripts["ensembl_gene_id"])]

# %%
sc.pp.scrublet(adata)

# %%
adata.obs["doublet_score"].plot(kind="hist")
adata.obs["doublet"] = adata.obs["doublet_score"] > 0.2

# %%
print(adata.obs.shape[0])
adata = adata[~adata.obs["doublet"].astype(bool)]
print(adata.obs.shape[0])

# %%
size_factor = np.median(np.array(adata.X.sum(1)))
adata.uns["size_factor"] = size_factor

# %%
adata.raw = adata

# %%
sc.pp.normalize_total(adata, target_sum=size_factor)
sc.pp.log1p(adata)
sc.pp.pca(adata)
sc.pp.highly_variable_genes(adata)

sc.pp.neighbors(adata)
sc.tl.umap(adata)

# %%
adata.layers["normalized"] = adata.X
adata.layers["counts"] = adata.raw.X

# %%
eyck.modalities.transcriptome.plot_umap(
    adata, ["Adamtsl2", "Alb", "Spp1", "Lyve1", "Stab2", "Dll4", "Wnt2", "Clec4f"]
).display()

# %%
magic_operator = magic.MAGIC(knn=30, solver="approximate")
X_smoothened = magic_operator.fit_transform(adata.X)
adata.layers["magic"] = X_smoothened

# %%
pickle.dump(adata, (folder_data_preproc / "adata.pkl").open("wb"))

# %% [markdown]
# ## Interpret and subset

# %%
adata = pickle.load((folder_data_preproc / "adata.pkl").open("rb"))

# %%
sc.tl.leiden(adata, resolution=1.0)

# %%
eyck.modalities.transcriptome.plot_umap(adata, color="leiden").display()

# %%
sc.tl.rank_genes_groups(
    adata, "leiden", method="wilcoxon", key_added="wilcoxon", use_raw=False
)
# genes_oi = sc.get.rank_genes_groups_df(adata, group=None, key='rank_genes_groups').sort_values("scores", ascending = False).groupby("group").head(2)["names"]

# %%
diffexp = (
    sc.get.rank_genes_groups_df(adata, group=None, key="wilcoxon")
    .sort_values("scores", ascending=False)
    .groupby("group")
    .head(5)
)
diffexp["symbol"] = diffexp["names"].apply(lambda x: adata.var.loc[x, "symbol"])
diffexp.set_index("group").loc["3"]

# %%
import io

marker_annotation = pd.read_table(
    io.StringIO(
        """ix	symbols	celltype
0	Alb, Aox3, Cyp2f2	Portal Hepatocyte
0	Sult2a8	Mid Hepatocyte
0	Slc1a2, Glul	Central Hepatocyte
0	Ptprb, Stab2	LSEC
0	Vwf	EC
1	Clec4f	KC
1	Lrat, Ngfr, Lama1, Ptger2	Stellate
1	Pkhd1, Egfr, Il1r1	Cholangiocyte
1	Ptprc	Immune
"""
    )
).set_index("celltype")
marker_annotation["symbols"] = marker_annotation["symbols"].str.split(", ")

# %%
import chromatinhd.utils.scanpy

cluster_celltypes = chromatinhd.utils.scanpy.evaluate_partition(
    adata, marker_annotation["symbols"].to_dict(), "symbol", partition_key="leiden"
).idxmax()

adata.obs["celltype"] = adata.obs["celltype"] = cluster_celltypes[
    adata.obs["leiden"]
].values
adata.obs["celltype"] = adata.obs["celltype"] = adata.obs["celltype"].astype(str)

# %%
eyck.modalities.transcriptome.plot_umap(
    adata, ["celltype", *marker_annotation["symbols"].explode()]
).display()
# sc.pl.umap(adata, color=["celltype", "leiden"], legend_loc="on data")

# %%
pickle.dump(adata, (folder_data_preproc / "adata_annotated.pkl").open("wb"))

# %% [markdown]
# ## TSS

# %%
adata = pickle.load((folder_data_preproc / "adata_annotated.pkl").open("rb"))

# %%
transcripts = pickle.load((folder_data_preproc / "transcripts.pkl").open("rb"))
transcripts = transcripts.loc[transcripts["ensembl_gene_id"].isin(adata.var.index)]

# %%
fragments_file = folder_data_preproc / "atac_fragments.tsv.gz"
selected_transcripts = chd.data.regions.select_tss_from_fragments(
    transcripts, fragments_file
)

# %%
np.log(transcripts.groupby("ensembl_gene_id")["n_fragments"].max()).shape

# %%
plt.scatter(
    adata.var["means"],
    np.log(transcripts.groupby("ensembl_gene_id")["n_fragments"].max())[
        adata.var.index
    ],
)
np.corrcoef(
    adata.var["means"],
    np.log(transcripts.groupby("ensembl_gene_id")["n_fragments"].max() + 1)[
        adata.var.index
    ],
)

# %%
pickle.dump(
    selected_transcripts, (folder_data_preproc / "selected_transcripts.pkl").open("wb")
)

# %% [markdown]
# ## Preprocess

# %%
dataset_folder = chd.get_output() / "datasets" / dataset_name
dataset_folder.mkdir(exist_ok=True, parents=True)

# %%
adata = pickle.load((folder_data_preproc / "adata_annotated.pkl").open("rb"))
adata.obs["cell_original"] = adata.obs.index
adata.obs.index = adata.obs.index.str.split("-").str[0] + "-1"

# %% [markdown]
# ### Create transcriptome

# %%
transcriptome = eyck.modalities.transcriptome.Transcriptome.from_adata(
    adata[:, adata.var.sort_values("dispersions_norm").tail(10000).index],
    path=dataset_folder / "transcriptome",
)

# %%
eyck.modalities.transcriptome.plot_umap(
    transcriptome, ["Dll4", "Wnt9b", "Wnt2", "Vwf", "Glul"], layer="magic"
).display()

# %% [markdown]
# ### 100k

# %%
selected_transcripts = pickle.load(
    (folder_data_preproc / "selected_transcripts.pkl").open("rb")
).loc[transcriptome.var.index]
regions = eyck.modalities.regions.Regions.from_transcripts(
    selected_transcripts, [-100000, 100000], dataset_folder / "regions" / "100k100k"
)

# %%
fragments_file = folder_data_preproc / "atac_fragments.tsv.gz"
fragments = eyck.modalities.Fragments(dataset_folder / "fragments" / "100k100k")
fragments.regions = regions
fragments = eyck.modalities.Fragments.from_fragments_tsv(
    fragments_file=fragments_file,
    regions=regions,
    obs=transcriptome.obs,
    path=fragments.path,
    overwrite=True,
)

# %%
fragments.create_regionxcell_indptr()

# %% [markdown]
# ## Plots

# %%
import pathlib
plot_folder = pathlib.Path("/home/wouters/fs4/u_mgu/private/wouters/grants/2024_ERC_AdG_Martin/multiome_vs_minibulk")
plot_folder.mkdir(exist_ok=True, parents=True)

# %%
dataset_folder = chd.get_output() / "datasets" / dataset_name

# %%
adata = pickle.load((folder_data_preproc / "adata_annotated.pkl").open("rb"))
transcriptome.adata.obs.loc[transcriptome.adata.obs["leiden"] == "12", "celltype"] = "Central Hepatocyte"

# %%
sc.pl.umap(transcriptome.adata, color = ["leiden", "celltype"], legend_loc = "on data")

# %%
eyck.modalities.transcriptome.plot_umap(
    adata, ["Dll4", "Wnt9b", "Wnt2", "Vwf", "Glul",]
).display()

# %%
adata.obs.loc[adata.obs["leiden"].isin(["12", "19"]), "celltype"] = "Central Hepatocyte"

sc.pl.umap(adata, color = ["celltype"])

# %%
import polyptich
adata2 = adata[adata.obs["celltype"].isin(["LSEC"])]
sc.pp.highly_variable_genes(adata2)
sc.pp.pca(adata2)
sc.pp.neighbors(adata2)
sc.tl.umap(adata2)

# %%
start = np.array([-5, 14])
adata2.obs["traj"] = np.sqrt((((adata2.obsm["X_umap"]) - start)**2).sum(1))
adata2.obs["traj"] = np.clip(adata2.obs["traj"] / np.quantile(adata2.obs["traj"], 0.98), 0, 1) + np.random.rand(adata2.shape[0]) * 0.2

plt.scatter(adata2.obsm["X_umap"][:, 0], adata2.obsm["X_umap"][:, 1], c = adata2.obs["traj"], cmap = "viridis")


# %%
zonation_cmap = mpl.colormaps["RdYlBu"]

fig = polyptich.grid.Figure(polyptich.grid.Wrap(ncol=2, padding_width = 0.1, padding_height = 0.2))

w = 1.0
h = 0.6
panel, ax = fig.main.add(polyptich.grid.Panel((w, h)))

plotdata = pd.DataFrame({
    "x":adata.obsm["X_umap"][:, 0],
    "y":adata.obsm["X_umap"][:, 1],
    "celltype":pd.Categorical(adata.obs["celltype"]),
}
)

celltype_colors = {
    "Central Hepatocyte":"#EC904F",
    "Portal Hepatocyte":"#EC904F",
    "Mid Hepatocyte":"#EC904F",
    "LSEC":"#137DE4",
    "KC":"#C12F40",
    "Stellate":"#EBCD48",
    "EC":"#19D2DB",
    "Immune":"#FF8392",
    "Cholangiocyte":"#FF8392",

}
plotdata["color"] = plotdata["celltype"].map(celltype_colors)

ax.scatter(plotdata["x"], plotdata["y"], c = plotdata["color"], cmap = "tab20", s = 1, lw = 0)
ax.axis("off")

for symbol in ["traj", "Dll4", "Ntn4", "Kit", "Wnt2"]:
    plotdata = pd.DataFrame({
        "x":adata2.obsm["X_umap"][:, 0],
        "y":adata2.obsm["X_umap"][:, 1],
    }
    )
    if symbol in ["traj", "distance_from_vein"]:
        plotdata["expression"] = adata2.obs[symbol].values
    else:
        plotdata["expression"] = sc.get.obs_df(adata2, eyck.modalities.transcriptome.gene_id(adata2.var, symbol)).values
    panel, ax = fig.main.add(polyptich.grid.Panel((w, h)))

    plotdata["expression"] = np.clip(plotdata["expression"] / np.quantile(plotdata["expression"], 0.99), 0, 1)

    plotdata = plotdata.sort_values("expression")

    if symbol in ["traj"]:
        cmap = zonation_cmap

    elif symbol in ["distance_from_vein"]:
        cmap = mpl.cm.viridis
    else:
        cmap = mpl.cm.YlGnBu

    ax.scatter(plotdata["x"], plotdata["y"], c = plotdata["expression"], cmap = cmap, s = 1. if symbol in ["traj", "distance_from_vein"] else 2, lw = 0)
    ax.axis("off")
    # ax.set_aspect("equal")
    # if symbol not in ["traj", "distance_from_vein"]:
    if symbol == "distance_from_vein":
        label = ""
    elif symbol == "traj":
        label = ""
    else:
        label = symbol
    ax.annotate(label, (0.5, 0.99), xycoords = "axes fraction", ha = "center", va = "bottom", fontsize = 8, fontstyle='italic')
# panel, ax = fig.main.add(polyptich.grid.Panel((1, 0.1)))

fig.display()

fig.savefig(plot_folder / "lsec_markers.pdf", dpi = 300)

# %%
