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

# splitter = "permutations_5fold5repeat"
# promoter_name, window = "100k100k", np.array([-100000, 100000])
# prediction_name = "v20_initdefault"

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
# ## Pairwindow

genes_all = transcriptome.var.index
# genes_all = transcriptome.var.query("symbol in ['BCL2']").index

scores = []
for gene in tqdm.tqdm(genes_all):
    scores_folder = prediction.path / "scoring" / "pairwindow_gene" / gene
    interaction_file = scores_folder / "interaction.pkl"

    if interaction_file.exists():
        scores.append(pd.read_pickle(interaction_file).assign(gene=gene).reset_index())
scores = pd.concat(scores)
print(len(scores["gene"].unique()))

# %%
scores["distance"] = np.abs(scores["window1"] - scores["window2"])
scores_oi = scores.query("distance > 1000").query("qval < 0.1")

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
scores_significant = scores.query("distance > 1000").query("qval < 0.1")
scores_significant.query("gene in @genes_oi").groupby("gene").first().sort_values(
    "cor", ascending=False
).head(10).assign(symbol=lambda x: transcriptome.var.loc[x.index, "symbol"].values)

# %%
bins = np.linspace(0, (window[1] - window[0]), 50)
synergism_distance = pd.DataFrame(
    {
        "total": np.bincount(
            np.digitize(
                scores.query("distance > 1000")["distance"],
                bins,
            ),
            minlength=len(bins),
        ),
        "synergistic": np.bincount(
            np.digitize(
                scores_significant["distance"],
                bins,
            ),
            minlength=len(bins),
        ),
        "bin": bins,
    }
)
synergism_distance["total"] = (
    synergism_distance["total"] / synergism_distance["total"].sum()
)
synergism_distance["synergistic"] = (
    synergism_distance["synergistic"] / synergism_distance["synergistic"].sum()
)

# %%
fig, ax = plt.subplots(figsize=(2, 2))
ax.plot(
    synergism_distance["bin"],
    synergism_distance["total"],
    color="black",
)

ax.plot(
    synergism_distance["bin"],
    synergism_distance["synergistic"],
    color="black",
)

# %%
bins = np.linspace(*window, 20)
synergism_position = pd.DataFrame(
    {
        "total": np.bincount(
            np.digitize(
                scores.query("distance > 1000")[
                    ["window1", "window2"]
                ].values.flatten(),
                bins,
            ),
            minlength=len(bins),
        ),
        "synergistic": np.bincount(
            np.digitize(
                scores_significant[["window1", "window2"]].values.flatten(),
                bins,
            ),
            minlength=len(bins),
        ),
        "bin": bins,
    }
)
synergism_position["total"] = (
    synergism_position["total"] / synergism_position["total"].sum()
)
synergism_position["synergistic"] = (
    synergism_position["synergistic"] / synergism_position["synergistic"].sum()
)

# %%
fig, ax = plt.subplots(figsize=(2, 2))
ax.plot(
    synergism_position["bin"],
    synergism_position["total"],
    color="black",
)

ax.plot(
    synergism_position["bin"],
    synergism_position["synergistic"],
    color="orange",
)

# %% [markdown]
# ## Specific gene

# %%
# symbol = "CD74"
# symbol = "COBLL1"
# symbol = "BCL2"
# symbol = "CCR6"
symbol = "LYN"
# symbol = "MGAT5"
# symbol = "SPIB"
# symbol = "MGAT5"
# symbol = "CD22"
# symbol = "TNFRSF13C"
# symbol = "HLA-DMB"
symbol = "MEF2C"
symbol = "IL1B"
symbol = "MEF2C"
gene = transcriptome.var.query("symbol == @symbol").index[0]

# gene = "ENSG00000186431"
# symbol = transcriptome.var.loc[gene, "symbol"]
print(symbol)

# %%
promoter = promoters.loc[gene]
promoter_str = f"{promoter.chr}:{promoter.start}-{promoter.end}"
promoter_str

# %%
import cooler

# c = cooler.Cooler("4DNFIXVAKX9Q.mcool::/resolutions/1000")

# !ln -s ~/NAS2/wsaelens/projects/chromatinhd/chromatinhd_manuscript/output/4DNFIXP4QG5B.mcool ~/projects/chromatinhd/chromatinhd_manuscript/output/4DNFIXP4QG5B.mcool
hic, bins_hic = chdm.hic.extract_hic(promoter)


# %%
scores_oi = scores.query("gene == @gene").copy()
assert len(scores_oi) > 0

# %%
scores_oi["hicwindow1"] = chdm.hic.match_windows(scores_oi["window1"].values, bins_hic)
scores_oi["hicwindow2"] = chdm.hic.match_windows(scores_oi["window2"].values, bins_hic)

# %%
matching_chd = (
    scores_oi
    # .query("qval < 0.2")
    .groupby(["hicwindow1", "hicwindow2"])["cor"]
    .max()
    .reset_index()
    .rename(columns={"hicwindow1": "window1", "hicwindow2": "window2"})
    .set_index(["window1", "window2"])
)

# %%
matching = chdm.hic.create_matching(bins_hic, scores_oi, hic)


# %%
for distance in [1000, 2000, 3000, 4000, 5000, 6000]:
    print(
        matching.query("distance > @distance")[["balanced", "cor"]]
        .corr()
        .loc["balanced", "cor"]
    )
    if distance == 3000:
        fig, ax = plt.subplots()
        ax.scatter(
            matching.query("distance > @distance")["balanced"],
            matching.query("distance > @distance")["cor"],
            alpha=0.1,
        )
        ax.set_xlabel("Hi-C")
        ax.set_ylabel("ChromatinHD")

# %%
import scipy.stats

matching_oi = matching.query("distance > 2000")

contingency = pd.crosstab(
    matching_oi["balanced"] > matching_oi["balanced"].mean(),
    matching_oi["cor"] > matching_oi["cor"].mean(),
)
# contingency = pd.crosstab(
#     matching_oi["balanced"] > 0,
#     matching_oi["cor"] > 0.05,
# )
print(contingency)
scipy.stats.chi2_contingency(contingency)

# %%
print(
    (contingency.iloc[0, 1] / contingency.iloc[:, 1].sum()),
    (contingency.iloc[1, 0] / contingency.iloc[1, :].sum()),
)
(contingency.iloc[0, 1] / contingency.iloc[:, 1].sum()) > (
    contingency.iloc[1, 0] / contingency.iloc[1, :].sum()
)

# %%
import scipy.stats

# %%
lm = scipy.stats.linregress(
    matching_oi["balanced"],
    matching_oi["cor"],
)
residuals = matching_oi["cor"] - (lm.slope * matching_oi["balanced"] + lm.intercept)

# %%
lm_residuals = scipy.stats.linregress(
    matching_oi["balanced"],
    np.sqrt(residuals**2),
)
(lm.rvalue, lm_residuals.slope, lm_residuals.pvalue)

# %%
main = chd.grid.Grid(padding_width=0)
fig = chd.grid.Figure(main)

plotdata = matching.query("distance > 0")
plotdata = matching.query("distance > 2000")

panel_hic = main[0, 0] = chd.grid.Panel((2, 2))
ax = panel_hic.ax
ax.matshow(
    np.log1p(plotdata.groupby(["window1", "window2"])["balanced"].max().unstack())
)
ax.set_title("Hi-C (GM12878)")

panel_chd = main[0, 1] = chd.grid.Panel((2, 2))
ax = panel_chd.ax
ax.matshow(plotdata.groupby(["window1", "window2"])["cor"].max().unstack())
ax.set_title("ChromatinHD co-predictivity (PBMCs)")
ax.set_yticks([])

fig.plot()

# %%
import scanpy as sc

sc.pl.umap(transcriptome.adata, color=["celltype", gene])

# %%
sc.tl.rank_genes_groups(
    transcriptome.adata,
    groupby="celltype",
)

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
    .query("cor > 0.4")
    .sort_values("cor", ascending=False)
    .index
)

# %%
# load or create gene hics
import pickle
import pathlib

if not pathlib.Path("gene_hics.pkl").exists():
    gene_hics = {}
    c = cooler.Cooler(
        str(chd.get_output() / "4DNFIXP4QG5B.mcool") + "::/resolutions/1000"
    )
    for gene in tqdm.tqdm(genes_all):
        promoter = promoters.loc[gene]
        promoter_str = f"{promoter.chr}:{promoter.start}-{promoter.end}"

        import cooler

        hic, bins_hic = chdm.hic.extract_hic(promoter, c=c)

        gene_hics[gene] = (hic, bins_hic)
    pickle.dump(gene_hics, open("gene_hics.pkl", "wb"))
else:
    gene_hics = pickle.load(open("gene_hics.pkl", "rb"))


# %%
scores_by_gene = {k: v for k, v in scores.groupby("gene")}


# %%
genescores = []

for gene in tqdm.tqdm(genes_oi):
    scores_folder = prediction.path / "scoring" / "pairwindow_gene" / gene
    interaction_file = scores_folder / "interaction.pkl"

    if gene in scores_by_gene:
        scores_oi = scores_by_gene[gene]
    elif interaction_file.exists():
        scores_oi = pd.read_pickle(interaction_file).assign(gene=gene).reset_index()
    else:
        continue

    promoter = promoters.loc[gene]
    promoter_str = f"{promoter.chr}:{promoter.start}-{promoter.end}"

    hic, bins_hic = gene_hics[gene]

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

    import scipy.stats

    matching_oi = matching.query("distance > 2000")
    lm = scipy.stats.linregress(
        matching_oi["balanced"],
        matching_oi["cor"],
    )
    residuals = matching_oi["cor"] - (lm.slope * matching_oi["balanced"] + lm.intercept)

    lm_residuals = scipy.stats.linregress(
        matching_oi["balanced"],
        np.sqrt(residuals**2),
    )

    genescores.append(
        {
            "gene": gene,
            "symbol": transcriptome.symbol(gene),
            "rvalue": lm.rvalue,
            "slope": lm_residuals.slope,
            "pvalue": lm_residuals.pvalue,
        }
    )
genescores = pd.DataFrame(genescores).set_index("gene")

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
genescores.sort_values("rvalue", ascending=False).head(10)
# %%
