# ---
# jupyter:
#   jupytext:
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

# %%
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

import torch_scatter
import torch

import tqdm.auto as tqdm

device = "cuda:0"

# %%
import peakfreeatac as pfa
import peakfreeatac.data

# %%
folder_root = pfa.get_output()
folder_data = folder_root / "data"

dataset_name = "lymphoma"
dataset_name = "pbmc10k"
# dataset_name = "e18brain"
# dataset_name = "pbmc3k-pbmc10k"
# dataset_name = "lymphoma+pbmc10k"
# dataset_name = "lymphoma-pbmc10k"
folder_data_preproc = folder_data / dataset_name

# %%
promoter_name, window = "10k10k", np.array([-10000, 10000])

# %%
transcriptome = peakfreeatac.data.Transcriptome(folder_data_preproc / "transcriptome")
fragments = peakfreeatac.data.Fragments(folder_data_preproc / "fragments" / promoter_name)

# %%
starts = np.arange(-200, 180, 10)[:]
ends = starts + 30
mids = starts + (ends - starts)/2

# %%
prediction_design = pd.DataFrame({"model_name":["v4_titr" + str(i) for i in starts] + ["v4", "v4_dummy", "v4_nn", "v4_1k-150-0_nn"], "mid":mids.tolist() + [pd.NA, pd.NA, pd.NA, pd.NA]})
prediction_design["dataset_name"] = dataset_name
prediction_design.index = prediction_design["model_name"]
prediction_design.index.name = "run"

# %%
prediction_design.shape


# %%
class Prediction(pfa.flow.Flow):
    pass


# %%
scores = {}
genescores = {}
for run_name, design_row in prediction_design.iterrows():
    prediction = Prediction(pfa.get_output() / "prediction_sequence" / design_row["dataset_name"] / promoter_name / design_row["model_name"])
    
    scores_dir = (prediction.path / "scoring" / "overall")
    
    scores_prediction = pd.read_pickle(scores_dir / "scores.pkl")
    genescores_prediction = pd.read_pickle(scores_dir / "genescores.pkl")
    
    scores[run_name] = scores_prediction
    genescores[run_name] = genescores_prediction
scores = pd.concat(scores, names = ["run", "phase"])
genescores = pd.concat(genescores, names = ["run", "phase", "gene"])

scores = scores.join(prediction_design)

# %%
# scores = scores.join((scores - scores.loc["v4_dummy"]).rename(columns = lambda x: x+ "_relative"))

# %%
# genescores = genescores.join((genescores - genescores.loc["v4_dummy"]).rename(columns = lambda x: x+ "_relative"))
genescores["label"] = pd.Categorical(transcriptome.symbol(genescores.index.get_level_values("gene")).values)

# %%
plotdata = scores.xs("validation", level = "phase")
plotdata_main = plotdata.loc[~pd.isnull(plotdata["mid"])]

fig, ax = plt.subplots()
ax.axvline(0, dashes = (2, 2), color = "#333333")
ax.plot(plotdata_main["mid"], plotdata_main["cor"])
ax.plot(-plotdata_main["mid"], plotdata_main["cor"], color = "grey", alpha = 0.1)
ax.axhline(plotdata.loc["v4_1k-150-0_nn"]["cor"], dashes = (3, 3), color = "purple")
ax.axhline(plotdata.loc["v4_nn"]["cor"], dashes = (3, 3), color = "cyan")
ax.axhline(plotdata.loc["v4"]["cor"], dashes = (3, 3), color = "blue")
ax.axhline(plotdata.loc["v4_dummy"]["cor"], dashes = (3, 3), color = "tomato")

# %% [markdown]
# ### Investigate linear model

# %%
import pickle
prediction = Prediction(pfa.get_output() / "prediction_sequence_v5" / dataset_name / promoter_name / "v4")

# %%
motifscan_folder = pfa.get_output() / "motifscans" / dataset_name / promoter_name
motifscan_folder.mkdir(parents=True, exist_ok=True)
folder_motifs = pfa.get_output() / "data" / "motifs" / "hs" / "hocomoco"

# %%
pwms = pickle.load((folder_motifs / "pwms.pkl").open("rb"))
motifs = pickle.load((motifscan_folder / "motifs.pkl").open("rb"))

# %%
motif_linear_scores = {}
prediction_design_oi = prediction_design.loc[(~pd.isnull(prediction_design["mid"])) | (prediction_design.index == "v4")]
for run_name, design_row in prediction_design_oi.iterrows():
    prediction = Prediction(pfa.get_output() / "prediction_sequence" / design_row["dataset_name"] / promoter_name / design_row["model_name"])
    for i in range(5):
        model = pickle.load(open(prediction.path / ("model_" + str(0) + ".pkl"), "rb"))
        motif_linear_scores_model = pd.Series(
            model.embedding_gene_pooler.nn1[0].weight.detach().cpu().numpy()[0],
            index = motifs.index
        ).sort_values(ascending = False)
        
        motif_linear_scores[(run_name, i)] = motif_linear_scores_model
motif_linear_scores = pd.concat(motif_linear_scores, names = ["run", "model_ix", "motif"])
motif_linear_scores = motif_linear_scores.groupby(["run", "motif"]).mean()

# %% [markdown]
# #### Correlation between weights

# %%
model_cors = pd.DataFrame(np.corrcoef(motif_linear_scores.unstack()), index = motif_linear_scores.unstack().index, columns = motif_linear_scores.unstack().index)

# %%
run_order = prediction_design.loc[~pd.isnull(prediction_design["mid"])].sort_values("mid").index

# %%
# sns.heatmap(model_cors.loc[run_order, run_order], vmin = -1, vmax = 1, cmap = mpl.cm.PiYG)
sns.heatmap(model_cors.loc[run_order, run_order], vmin = -1, vmax = 1, cmap = mpl.cm.PiYG)

# %% [markdown]
# Correlation between only positive/negative motifs

# %%
motifs_oi = motifs.loc[motif_linear_scores.loc["v4"] > 0.2]
# motifs_oi = motifs.loc[motif_linear_scores.loc["v4"] > -0.2]

# %%
motif_linear_scores_unstacked = motif_linear_scores.unstack()[motifs_oi.index]

# %%
model_cors = pd.DataFrame(np.corrcoef(motif_linear_scores_unstacked), index = motif_linear_scores_unstacked.index, columns = motif_linear_scores_unstacked.index)
run_order = prediction_design.loc[~pd.isnull(prediction_design["mid"])].sort_values("mid").index
# sns.heatmap(model_cors.loc[run_order, run_order], vmin = -1, vmax = 1, cmap = mpl.cm.PiYG)
sns.heatmap(model_cors.loc[run_order, run_order], vmin = -1, vmax = 1, cmap = mpl.cm.PiYG)

# %% [markdown]
# #### Difference in linear weight between full model and local models

# %%
motif_linear_scores_diff = (((motif_linear_scores - motif_linear_scores.loc["v4"]) ** 2).groupby("motif").mean()).sort_values()
motif_linear_scores_std = motif_linear_scores.groupby("motif").std()

# %%
import adjustText

# %%
fig, (ax0, ax1) = plt.subplots(1, 2, figsize = (10, 5))
plotdata = pd.DataFrame({"weight":motif_linear_scores.loc["v4"], "weight_diff":motif_linear_scores_diff})
ax0.scatter(
    plotdata["weight"],
    plotdata["weight_diff"]
)

y_col = "weight_diff"
motifs_oi = pd.DataFrame({"motif":plotdata["weight_diff"].sort_values(ascending = False).index[:10]}).set_index("motif")
texts = []
for motif in motifs_oi.index:
    x, y = plotdata.loc[motif, "weight"], plotdata.loc[motif, y_col]
    ax0.scatter(x, y, color = "red")
    texts.append(ax0.text(x, y, s = motifs.loc[motif, "gene_label"]))
adjustText.adjust_text(texts, ax = ax0)

plotdata = pd.DataFrame({"weight":motif_linear_scores.loc["v4"], "weight_std":motif_linear_scores_var})
ax1.scatter(
    plotdata["weight"],
    plotdata["weight_std"]
)

y_col = "weight_std"
motifs_oi = pd.DataFrame({"motif":plotdata["weight_std"].sort_values(ascending = False).index[:30]}).set_index("motif")
texts = []
for motif in motifs_oi.index:
    x, y = plotdata.loc[motif, "weight"], plotdata.loc[motif, y_col]
    ax1.scatter(x, y, color = "red")
    texts.append(ax1.text(x, y, s = motifs.loc[motif, "gene_label"]))
adjustText.adjust_text(texts, ax = ax1)

# %%
motif_linear_scores_var.sort_values(ascending = False).head(20)

# %%
plotdata = motif_linear_scores.to_frame("weight").join(prediction_design_oi)
motifs_oi = motifs.index[motifs.index.str.startswith("SALL4")]
plotdata = plotdata.query("motif in @motifs_oi")

# %%
fig, ax = plt.subplots()
for motif, plotdata_motif in plotdata.groupby("motif"):
    plotdata_motif = plotdata_motif.sort_values("mid")
    plotdata_motif = plotdata_motif.loc[~pd.isnull(plotdata_motif["mid"])]
    f = ax.plot(plotdata_motif["mid"], plotdata_motif["weight"], label = motif)
    ax.axhline(motif_linear_scores.loc["v4"][motif], color = f[0].get_color())
plt.legend()

# %%
motif_linear_scores.plot(kind = "hist")

# %%
