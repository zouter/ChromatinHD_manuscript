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
import os

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

dataset_name = "dezuani_2024"
genome = "GRCh38"
organism = "hs"

folder_data_preproc = folder_data / dataset_name
folder_data_preproc.mkdir(exist_ok=True, parents=True)

folder_dataset = chd.get_output() / "datasets" / dataset_name

# %% [markdown]
# ## Download

# %%
# Download overview
os.system(f"wget https://www.ebi.ac.uk/biostudies/files/E-MTAB-13070/E-MTAB-13070.sdrf.txt -O {folder_data_preproc}/E-MTAB-13070.sdrf.txt")

# %%
# overview.iloc[0].to_dict()

# %%
# (overview["Characteristics[individual]"] == "H_6").sum()

# %%

to_download = [
    "H_6_A-filtered_feature_bc_matrix.h5",
    "H_6_A-atac_fragments.tsv.gz",
    "H_5_A-filtered_feature_bc_matrix.h5",
    "H_5_B-filtered_feature_bc_matrix.h5",
    "H_7_A-filtered_feature_bc_matrix.h5",
    "H_7_B-filtered_feature_bc_matrix.h5",
    "H_7_C-filtered_feature_bc_matrix.h5",
]

main_url = "https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-/070/E-MTAB-13070/Files/"

for filename in to_download:
    if not (folder_data_preproc / filename).exists():
        import os
        os.system(f"wget {main_url}/{filename} -O {folder_data_preproc}/{filename}")


# %%
if not (folder_data_preproc / "41586_2024_7946_MOESM3_ESM.xlsx").exists():
    os.system(f"wget https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07946-4/MediaObjects/41586_2024_7946_MOESM3_ESM.xlsx -O {folder_data_preproc}/41586_2024_7946_MOESM3_ESM.xlsx")
markers = pd.read_excel(folder_data_preproc / "41586_2024_7946_MOESM3_ESM.xlsx", sheet_name = "Suppl. Tab 3", header = 2)

# %%
adatas = []
for sample_name, filename in [
    # ["H_6_A", "H_6_A-filtered_feature_bc_matrix.h5"],
    # ["H_5_A", "H_5_A-filtered_feature_bc_matrix.h5"],
    # ["H_5_B", "H_5_B-filtered_feature_bc_matrix.h5"],
    ["H_7_A", "H_7_A-filtered_feature_bc_matrix.h5"],
    ["H_7_B", "H_7_B-filtered_feature_bc_matrix.h5"],
    ["H_7_C", "H_7_C-filtered_feature_bc_matrix.h5"],
]:
    if not (folder_data_preproc / filename).exists():
        continue
    adata = sc.read_10x_h5(folder_data_preproc / filename)
    adata.obs["sample"] = sample_name
    adata.var_names_make_unique()
    adatas.append(adata)


# %%
adata = sc.concat(adatas, keys = [x.obs["sample"].values[0] for x in adatas])

# %%
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)

# %%
# adata.var["symbol"] = adata.var.index
# adata.var["gene"] = adata.var["gene_ids"]
# adata.var.index = adata.var["gene"]

# %%
for celltype in markers["Cell_type"].unique():
    sc.tl.score_genes(
        adata,
        np.unique(markers.query(f"Cell_type == '{celltype}'")["Names"].values),
        score_name = celltype,
    )

# %%
sc.tl.leiden(adata, resolution=1.0)

# %%
eyck.m.t.plot_umap(adata, color = ["sample", "leiden", *markers["Cell_type"].unique()]).display()

# %%
adata.var["symbol"] = adata.var.index
diffexp = eyck.m.t.diffexp.compare_all_groups(adata, "leiden")

# %%
diffexp.loc["LYZ"].query("pvals_adj<0.05")

# %%
diffexp.query("group == '4'").head(30)

# %%
eyck.m.t.plot_umap(adata, color = ["ITGAM", "ITGAX", "C1QA", "SLC40A1", "CD34"], datashader = False).display()
# eyck.m.t.plot_umap(adata, color = markers.query("Cell_type == 'Monocyte progenitors'")["Names"].unique()).display()

# %%
adata.var["PC_1"] = adata.varm["PCs"][:, 0]
adata.var["PC_2"] = adata.varm["PCs"][:, 1]
adata.var["PC_3"] = adata.varm["PCs"][:, 2]

# %%
gene_ids = [
    adata.var.index[np.argsort(adata.varm["PCs"][:, 0])][0],
    adata.var.index[np.argsort(adata.varm["PCs"][:, 1])][0],
    adata.var.index[np.argsort(adata.varm["PCs"][:, 2])][0],
    adata.var.index[np.argsort(adata.varm["PCs"][:, 3])][0],
    adata.var.index[np.argsort(adata.varm["PCs"][:, 4])][0],
    adata.var.index[np.argsort(adata.varm["PCs"][:, 5])][0],
    adata.var.index[np.argsort(-adata.varm["PCs"][:, 0])][0],
    adata.var.index[np.argsort(-adata.varm["PCs"][:, 1])][0],
    adata.var.index[np.argsort(-adata.varm["PCs"][:, 2])][0],
    adata.var.index[np.argsort(-adata.varm["PCs"][:, 3])][0],
    adata.var.index[np.argsort(-adata.varm["PCs"][:, 4])][0],
    adata.var.index[np.argsort(-adata.varm["PCs"][:, 5])][0],
]
eyck.m.t.plot_umap(adata, color = gene_ids).display()


# %% [markdown]
# The original data is not block compressed, so we need to recompress the data using tabix.

# %%
# # !zcat {folder_data_preproc}/atac_fragments.tsv.gz > {folder_data_preproc}/atac_fragments.tsv
print(f"zcat {folder_data_preproc}/atac_fragments.tsv.gz.raw > {folder_data_preproc}/atac_fragments.tsv")

# %%
if not (folder_data_preproc/"atac_fragments.tsv.gz").exists():
    import pysam
    pysam.tabix_compress(str(folder_data_preproc/"atac_fragments.tsv"), str(folder_data_preproc/"atac_fragments.tsv.gz"), force=True)
    pysam.tabix_index(str(folder_data_preproc/"atac_fragments.tsv.gz"), seq_col = 0, start_col = 1, end_col = 2)
    
# %%
if (folder_data_preproc/"atac_fragments.tsv").exists():
    (folder_data_preproc/"atac_fragments.tsv").unlink()
if (folder_data_preproc/"atac_fragments.tsv.gz.raw").exists():
    (folder_data_preproc/"atac_fragments.tsv.gz.raw").unlink()


# %%
print(f"wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE218nnn/GSE218468/suppl/GSE218468_multiome_rna_counts.tsv.gz -O {folder_data_preproc}/rna_counts.tsv.gz")

# %% [markdown]
# ## Create transcriptome

# %%
transcriptome = eyck.modalities.Transcriptome(folder_dataset / "transcriptome")

# %%
counts = pd.read_table(folder_data_preproc / "rna_counts.tsv.gz", index_col=0, sep = "\t").T

# %%
obs = pd.DataFrame({"cell":counts.index}).set_index("cell")
var = pd.DataFrame({"symbol":counts.columns})
mapping = chd.biomart.map_symbols(chd.biomart.Dataset.from_genome(genome), var["symbol"]).groupby("external_gene_name").first().reindex(var["symbol"])
mapping["ix"] = np.arange(len(mapping))
mapping = mapping.dropna()
var = var.loc[var["symbol"].isin(mapping.index)]
var.index = pd.Series(mapping["ensembl_gene_id"], name = "gene")
counts = counts.loc[:,var["symbol"]]

# %%
adata = sc.AnnData(scipy.sparse.csr_matrix(counts.values), obs = obs, var = var)

# %%
transcripts = chd.biomart.get_transcripts(chd.biomart.Dataset.from_genome(genome), gene_ids=adata.var.index.unique())
pickle.dump(transcripts, (folder_data_preproc / 'transcripts.pkl').open("wb"))

# %%
# only retain genes that have at least one ensembl transcript
adata = adata[:, adata.var.index.isin(transcripts["ensembl_gene_id"])]

# %%
sc.pp.scrublet(adata)

# %%
adata.obs["doublet_score"].plot(kind="hist")
adata.obs["doublet"] = (adata.obs["doublet_score"] > 0.1).astype("category")

print(adata.obs.shape[0])
adata = adata[~adata.obs["doublet"].astype(bool)]
print(adata.obs.shape[0])

# %%
size_factor = np.median(np.array(adata.X.sum(1)))
adata.uns["size_factor"] = size_factor

# %%
adata.raw = adata

# %%
sc.pp.normalize_total(adata, target_sum = size_factor)
sc.pp.log1p(adata)
sc.pp.pca(adata)
sc.pp.highly_variable_genes(adata)

sc.pp.neighbors(adata)
sc.tl.umap(adata)

# %%
adata.layers["normalized"] = adata.X
adata.layers["counts"] = adata.raw.X

# %%
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
sc.tl.leiden(adata, resolution=1.0)
sc.pl.umap(adata, color="leiden")

# %%
sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon", key_added  = "wilcoxon", use_raw = False)
# genes_oi = sc.get.rank_genes_groups_df(adata, group=None, key='rank_genes_groups').sort_values("scores", ascending = False).groupby("group").head(2)["names"]

# %%
diffexp = sc.get.rank_genes_groups_df(adata, group=None, key='wilcoxon').sort_values("scores", ascending = False).groupby("group").head(5)
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
1	Clec4f	KC
1	Rbms3	Stellate
1	Pkhd1, Egfr, Il1r1	Cholangiocyte
1	Ptprc	Immune
"""
    )
).set_index("celltype")
marker_annotation["symbols"] = marker_annotation["symbols"].str.split(", ")

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

# %%
eyck.modalities.transcriptome.plot_umap(adata, ["celltype", "Dll4", "Wnt2", "Wnt9b", "Glul", "Epcam", "Egfr", "Slc1a2", "Vwf", "Lyve1"]).display()
# sc.pl.umap(adata, color=["celltype", "leiden"], legend_loc="on data")

# %%
pickle.dump(adata, (folder_data_preproc / 'adata_annotated.pkl').open("wb"))

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
np.log(transcripts.groupby("ensembl_gene_id")["n_fragments"].max()).shape

# %%
plt.scatter(adata.var["means"], np.log(transcripts.groupby("ensembl_gene_id")["n_fragments"].max())[adata.var.index])
np.corrcoef(adata.var["means"], np.log(transcripts.groupby("ensembl_gene_id")["n_fragments"].max() + 1)[adata.var.index])

# %%
pickle.dump(selected_transcripts, (folder_data_preproc / 'selected_transcripts.pkl').open("wb"))

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
transcriptome = chd.data.transcriptome.Transcriptome.from_adata(adata[:, adata.var.sort_values("dispersions_norm").tail(5000).index], path=dataset_folder / "transcriptome")

# %%
sc.pl.umap(adata, color = adata.var.index[(adata.var["symbol"] == "Glul")][0])

# %% [markdown]
# ### 10k

# %%
selected_transcripts = pickle.load((folder_data_preproc / "selected_transcripts.pkl").open("rb")).loc[transcriptome.var.index]
regions = chd.data.regions.Regions.from_transcripts(
    selected_transcripts, [-10000, 10000], dataset_folder / "regions" / "10k10k"
)

# %%
# # !zcat {folder_data_preproc}/atac_fragments.tsv.gz | head -n 1000

# %%
fragments_file = folder_data_preproc / "atac_fragments.tsv.gz"
fragments = chd.data.Fragments(dataset_folder / "fragments" / "10k10k")

fragments.regions = regions
fragments = chd.data.Fragments.from_fragments_tsv(
    fragments_file=fragments_file,
    regions=regions,
    obs=transcriptome.obs,
    path=fragments.path,
    overwrite = True
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
    overwrite = True,
)

# %%
fragments.create_regionxcell_indptr()

# %% [markdown]
# ## Clustering

# %%
transcriptome = eyck.modalities.Transcriptome(folder_dataset / "transcriptome")

# %%
clustering = chd.data.Clustering.from_labels(
    transcriptome.obs["celltype"], path=folder_dataset / "clusterings" / "cluster"

)

# %% [markdown]
# ## Folds

# %%
folds = chd.data.folds.Folds(
    folder_dataset / "folds"
)
folds.folds = [
    {
        "cells_train": np.arange(len(transcriptome.obs)),
        "cells_test": np.arange(len(transcriptome.obs))[:500],
        "cells_validation": np.arange(len(transcriptome.obs))[:500],
    }
]