# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

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

import tqdm.auto as tqdm
import xarray as xr

# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

# transcriptome
# dataset_name = "lymphoma"
dataset_name = "pbmc10k"
# dataset_name = "pbmc10k_gran"
# dataset_name = "e18brain"
folder_data_preproc = folder_data / dataset_name

transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")

splitter = "random_5fold"
promoter_name, window = "10k10k", np.array([-10000, 10000])
prediction_name = "v20_initdefault"

splitter = "permutations_5fold5repeat"
promoter_name, window = "10k10k", np.array([-10000, 10000])
prediction_name = "v20"

splitter = "permutations_5fold5repeat"
promoter_name, window = "100k100k", np.array([-100000, 100000])
prediction_name = "v20_initdefault"

# fragments
promoters = pd.read_csv(
    folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
)
window_width = window[1] - window[0]

fragments = chd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)
fragments.obs.index.name = "cell"

# %%
print(prediction_name)
prediction = chd.flow.Flow(
    chd.get_output()
    / "prediction_positional"
    / dataset_name
    / promoter_name
    / splitter
    / prediction_name
)

# %% [markdown]
# ## Transcriptome diffexp

# %%
import scanpy as sc

sc.tl.rank_genes_groups(transcriptome.adata, groupby="celltype")
diffexp = (
    sc.get.rank_genes_groups_df(transcriptome.adata, group="naive B")
    .rename(columns={"names": "gene"})
    .assign(symbol=lambda x: transcriptome.var.loc[x["gene"], "symbol"].values)
    .set_index("gene")
)

# %%
genes_oi = diffexp.query("pvals_adj < 0.05").query("logfoldchanges > 0.5").index

# %%
if "scores_by_gene" not in globals():
    scores_by_gene = {}
# scores_by_gene = {k: v for k, v in scores.groupby("gene")}

# %% [markdown]
# ## For all genes

# %%
genes_all = transcriptome.var.index

scorer_folder = prediction.path / "scoring" / "nothing"
nothing_scoring = chd.scoring.prediction.Scoring.load(scorer_folder)
genes_oi = (
    nothing_scoring.genescores.mean("model")
    .sel(phase=["test", "validation"])
    .mean("phase")
    .sel(i=0)
    .to_pandas()
    .query("cor > 0.1")
    .sort_values("cor", ascending=False)
    .index
)

# %%
import pickle

cool_name = "rao_2014_1kb"
hic_file = folder_data_preproc / "hic" / promoter_name / f"{cool_name}.pkl"
gene_hics = pickle.load(hic_file.open("rb"))


# %%
genescores = []

import scipy.stats

for gene in tqdm.tqdm(genes_oi):
    score = {}

    score.update(
        {
            "gene": gene,
            "symbol": transcriptome.symbol(gene),
        }
    )

    scores_folder = prediction.path / "scoring" / "pairwindow_gene" / gene
    interaction_file = scores_folder / "interaction.pkl"

    if gene in scores_by_gene:
        scores_oi = scores_by_gene[gene]
    elif interaction_file.exists():
        scores_oi = pd.read_pickle(interaction_file).assign(gene=gene).reset_index()
        scores_by_gene[gene] = scores_oi
    else:
        continue

    promoter = promoters.loc[gene]
    promoter_str = f"{promoter.chr}:{promoter.start}-{promoter.end}"

    hic, bins_hic = gene_hics[gene]
    # hic, bins_hic = chdm.hic.fix_hic(hic, bins_hic)

    scores_oi["hicwindow1"] = chdm.hic.match_windows(
        scores_oi["window1"].values, bins_hic
    )
    scores_oi["hicwindow2"] = chdm.hic.match_windows(
        scores_oi["window2"].values, bins_hic
    )

    matching = chdm.hic.create_matching(
        bins_hic,
        scores_oi,
        # scores_oi.query("qval < 0.2"),
        hic,
    )

    matching_oi = matching.query("distance > 10000")
    lm = scipy.stats.linregress(
        matching_oi["balanced"],
        matching_oi["cor"],
    )

    score.update(
        {
            "rvalue": lm.rvalue,
        }
    )

    # maxpool
    # hic2 = maxpool_hic(hic, bins_hic)
    # matching2 = chdm.hic.create_matching(
    #     bins_hic,
    #     scores_oi,
    #     # scores_oi.query("qval < 0.2"),
    #     hic2,
    # )

    # matching2_oi = matching2.query("distance > 1000")
    # lm2 = scipy.stats.linregress(
    #     matching2_oi["balanced"],
    #     matching2_oi["cor"],
    # )

    # score.update(
    #     {
    #         "rvalue_maxpool": lm2.rvalue,
    #     }
    # )

    # # maxpool
    # hic2 = maxipool_hic(hic, bins_hic)
    # matching2 = chdm.hic.create_matching(
    #     bins_hic,
    #     scores_oi,
    #     # scores_oi.query("qval < 0.2"),
    #     hic2,
    # )

    # matching2_oi = matching2.query("distance > 1000")
    # lm2 = scipy.stats.linregress(
    #     matching2_oi["balanced"],
    #     matching2_oi["cor"],
    # )

    # score.update(
    #     {
    #         "rvalue_maxipool": lm2.rvalue,
    #     }
    # )

    # # maxpool
    # hic2 = maxipool_hic(hic, bins_hic, k=2)
    # matching2 = chdm.hic.create_matching(
    #     bins_hic,
    #     scores_oi,
    #     # scores_oi.query("qval < 0.2"),
    #     hic2,
    # )

    # matching2_oi = matching2.query("distance > 1000")
    # lm2 = scipy.stats.linregress(
    #     matching2_oi["balanced"],
    #     matching2_oi["cor"],
    # )

    # score.update(
    #     {
    #         "rvalue_maxipool_5": lm2.rvalue,
    #     }
    # )

    genescores.append(score)

    if len(genescores) > 100:
        break
genescores = pd.DataFrame(genescores).set_index("gene")

# %%
fig, ax = plt.subplots(figsize=(2, 2))
ax.hist(genescores["rvalue"], bins=20)

# %%
fig, ax = plt.subplots(figsize=(2, 2))
genes_diffexp_b = diffexp.query("pvals_adj < 0.05").query("logfoldchanges > 1.0").index
genes_diffexp_b = genes_diffexp_b[genes_diffexp_b.isin(genescores.index)]
genes_diffexp_a = [gene for gene in genescores.index if gene not in genes_diffexp_b]
sns.ecdfplot(data=genescores, x="rvalue", label="actual")
sns.ecdfplot(data=genescores, x="rvalue_maxpool", label="maxpool")
sns.ecdfplot(data=genescores, x="rvalue_maxipool", label="maxipool")
sns.ecdfplot(data=genescores, x="rvalue_maxipool_5", label="maxipool")

plt.legend()

# %%
(
    genescores["rvalue"].mean(),
    genescores["rvalue_maxpool"].mean(),
    genescores["rvalue_maxipool"].mean(),
    genescores["rvalue_maxipool_5"].mean(),
)

# %%
fig, ax = plt.subplots(figsize=(2, 2))
genes_diffexp_b = diffexp.query("pvals_adj < 0.05").query("logfoldchanges > 1.0").index
genes_diffexp_b = genes_diffexp_b[genes_diffexp_b.isin(genescores.index)]
genes_diffexp_a = [gene for gene in genescores.index if gene not in genes_diffexp_b]
sns.ecdfplot(data=genescores, x="rvalue")
sns.ecdfplot(
    data=genescores.loc[genes_diffexp_a],
    x="rvalue",
)
sns.ecdfplot(
    data=genescores.loc[genes_diffexp_b],
    x="rvalue",
)

# %%
genescores.sort_values("rvalue_maxpool", ascending=False).head(10)

# %%
genescores["rvalue"].mean()

# %%
matching = chdm.hic.create_matching(
    bins_hic,
    scores_oi,
    # scores_oi.query("qval < 0.2"),
    hic,
)

import scipy.stats

matching_oi = matching.query("distance >= 2000")
lm = scipy.stats.linregress(
    matching_oi["balanced"],
    matching_oi["cor"],
)
lm.rvalue
# %%
matching = chdm.hic.create_matching(
    bins_hic,
    scores_oi,
    # scores_oi.query("qval < 0.2"),
    hic2,
)

import scipy.stats

matching_oi = matching.query("distance >= 2000")
lm2 = scipy.stats.linregress(
    matching_oi["balanced"],
    matching_oi["cor"],
)
lm2.rvalue


# %%
genescores = []

import scipy.stats

for gene in tqdm.tqdm(genes_oi):
    score = {}

    score.update(
        {
            "gene": gene,
            "symbol": transcriptome.symbol(gene),
        }
    )

    hic, bins_hic = gene_hics[gene]
    hic, bins_hic = chdm.hic.fix_hic(hic, bins_hic)

    hic["distance"] = np.abs(
        hic.index.get_level_values("window1").astype(float)
        - hic.index.get_level_values("window2").astype(float)
    )


# %%
