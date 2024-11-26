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

dataset_name = "pbmc10k"
folder_data_preproc = folder_data / dataset_name

transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")

splitter = "5x1"
regions_name, window = "100k100k", np.array([-100000, 100000])
prediction_name = "v33"
layer = "magic"

fragments = chd.data.Fragments(chd.get_output() / "datasets" / dataset_name / "fragments" / regions_name)

# %%
print(prediction_name)
prediction = chd.flow.Flow(
    chd.get_output()
    / "pred"
    / dataset_name
    / regions_name
    / splitter
    / layer
    / prediction_name
)

# %%
regionmultiwindow = chd.models.pred.interpret.RegionMultiWindow(prediction.path / "scoring" / "regionmultiwindow")

# %%
import chromatinhd.data.motifscan

# %%
fasta_file = "/data/genome/GRCh38/GRCh38.fa"

# %%
regions_oi = fragments.regions.coordinates.loc[[transcriptome.gene_id("IRF1")]]

# %%
region_onehots = chd.data.motifscan.motifscan.create_region_onehots(None, fasta_file, regions_oi)

# %%
import torch

# %%
pfm = pd.read_table("Figure1A_Tn5_sepcat_bias.transfac")[["A", "C", "G", "T"]]
pwm = np.log2(pfm/pfm.sum(1).values[:, None] / 0.25)
pwm2 = np.log(pfm/pfm.sum(1).values[:, None] / 0.25)
pwm_torch = torch.tensor(pwm2.values).float().T

# %%
gene_oi = transcriptome.gene_id("IRF1")
region_ix = fragments.var.index.get_loc(gene_oi)

# %%
scores_original, positions, strand = chd.data.motifscan.motifscan.scan(
    region_onehots[list(region_onehots.keys())[0]].T[None, :, :], pwm_torch, cutoff = -99999999999.
)

padding = 20
fragments_oi = fragments.mapping[:, 1] == region_ix
coordinates_oi = fragments.coordinates[fragments_oi].flatten() - fragments.regions.window[0]
coordinates_oi = coordinates_oi[(coordinates_oi >= padding) & (coordinates_oi < fragments.regions.window_width - padding)]
bincounts = np.bincount(coordinates_oi, minlength = fragments.regions.window_width)

for l in range(-10, 10):
    scores = np.pad(scores_original.reshape(2, -1), ((0, 0), (15-l, 15+l)), "constant", constant_values = 0).max(0)
    
    print(l, np.corrcoef(scores, np.log1p(bincounts))[0, 1])

# %%
cells_oi_1 = transcriptome.obs["celltype"].isin(["CD14+ Monocytes", "FCGR3A+ Monocytes", "cDCs"]).values
cells_oi_2 = ~cells_oi_1

cells_oi_1 = np.where(cells_oi_1)[0]
cells_oi_2 = np.where(cells_oi_2)[0]

# %%
fragments_oi = (fragments.mapping[:, 1] == region_ix) & (np.isin(fragments.mapping[:, 0], cells_oi_1))
coordinates_oi = fragments.coordinates[fragments_oi].flatten() - fragments.regions.window[0]
padding = 30
coordinates_oi = coordinates_oi[(coordinates_oi >= padding) & (coordinates_oi < fragments.regions.window_width - padding)]

bincounts1 = np.bincount(coordinates_oi, minlength = fragments.regions.window_width)

fragments_oi = (fragments.mapping[:, 1] == region_ix) & (np.isin(fragments.mapping[:, 0], cells_oi_2))
coordinates_oi = fragments.coordinates[fragments_oi].flatten() - fragments.regions.window[0]
padding = 30
coordinates_oi = coordinates_oi[(coordinates_oi >= padding) & (coordinates_oi < fragments.regions.window_width - padding)]

bincounts2 = np.bincount(coordinates_oi, minlength = fragments.regions.window_width)

# %%
pseudo = 1.
# pseudo = 1e-5

lfc = np.log((bincounts1 + pseudo)/len(cells_oi_1) * 1e6) - np.log((bincounts2 + pseudo)/len(cells_oi_2) * 1e6) - (np.log(pseudo/len(cells_oi_1) * 1e6) - np.log(pseudo/len(cells_oi_2) * 1e6))
sns.ecdfplot(lfc)

# %%
l = -1
scores = np.pad(scores_original.reshape(2, -1), ((0, 0), (15-l, 15+l)), "constant", constant_values = 0).max(0)

# %%
plotdata = pd.DataFrame({
    "bincount":bincounts,
    "score":scores,
    "lfc":lfc
})
plotdata["abslfc"] = np.abs(plotdata["lfc"])
plotdata["log1p_bincount"] = np.log1p(plotdata["bincount"])

# %%
fig = polyptich.grid.Figure(polyptich.grid.Grid())

ylims = [-5, 5]

panel, ax = fig.main.add_right(polyptich.grid.Panel((2, 2)))
ax.scatter(plotdata["log1p_bincount"], plotdata["score"], s = 1)
ax.set_xlabel("log1p(bincounts)")
ax.set_ylabel("scores")
cor = np.corrcoef(plotdata["log1p_bincount"], plotdata["score"])[0, 1]
ax.annotate(f"cor = {cor:.2f}", (0.95, 0.05), xycoords = "axes fraction", ha = "right", va = "bottom")
ax.set_ylim(*ylims)

plotdata_oi = plotdata#[plotdata["log1p_bincount"] > 0.]

panel, ax = fig.main.add_right(polyptich.grid.Panel((2, 2)))
ax.scatter(plotdata_oi["abslfc"], plotdata_oi["score"], s = 1)
ax.set_xlabel("abs(lfc)")
ax.set_ylabel("scores")
cor = np.corrcoef(plotdata_oi["abslfc"], plotdata_oi["score"])[0, 1]
ax.annotate(f"cor = {cor:.2f}", (0.95, 0.05), xycoords = "axes fraction", ha = "right", va = "bottom")
ax.set_ylabel("")
ax.set_ylim(*ylims)
ax.set_title("All positions")

plotdata_oi = plotdata[plotdata["log1p_bincount"] > 0.]

panel, ax = fig.main.add_right(polyptich.grid.Panel((2, 2)))
ax.scatter(plotdata_oi["abslfc"], plotdata_oi["score"], s = 1)
ax.set_xlabel("abs(lfc)")
ax.set_ylabel("scores")
cor = np.corrcoef(plotdata_oi["abslfc"], plotdata_oi["score"])[0, 1]
ax.annotate(f"cor = {cor:.2f}", (0.95, 0.05), xycoords = "axes fraction", ha = "right", va = "bottom")
ax.set_ylabel("")
ax.set_ylim(*ylims)
ax.set_title("Positions with at least 1 cut")

plotdata_oi = plotdata[plotdata["log1p_bincount"] > np.log(5+1)]

panel, ax = fig.main.add_right(polyptich.grid.Panel((2, 2)))
ax.scatter(plotdata_oi["abslfc"], plotdata_oi["score"], s = 1)
ax.set_xlabel("abs(lfc)")
ax.set_ylabel("scores")
cor = np.corrcoef(plotdata_oi["abslfc"], plotdata_oi["score"])[0, 1]
ax.annotate(f"cor = {cor:.2f}", (0.95, 0.05), xycoords = "axes fraction", ha = "right", va = "bottom")
ax.set_ylabel("")
ax.set_ylim(*ylims)
ax.set_title("Positions with at least 10 cuts")


plotdata_oi = plotdata[plotdata["log1p_bincount"] > np.log(10+1)]

panel, ax = fig.main.add_right(polyptich.grid.Panel((2, 2)))
ax.scatter(plotdata_oi["abslfc"], plotdata_oi["score"], s = 1)
ax.set_xlabel("abs(lfc)")
ax.set_ylabel("scores")
cor = np.corrcoef(plotdata_oi["abslfc"], plotdata_oi["score"])[0, 1]
ax.annotate(f"cor = {cor:.2f}", (0.95, 0.05), xycoords = "axes fraction", ha = "right", va = "bottom")
ax.set_ylabel("")
ax.set_ylim(*ylims)
ax.set_title("Positions with at least 100 cuts")

fig.plot()

# %%
# rolling window
k = 25
for i in range()

# %%
((plotdata["score"] > 0.).rolling(25).mean().fillna(1.) > 0).sum()

# %%
fig, ax = plt.subplots()
sns.ecdfplot(np.diff(np.where((plotdata["score"] > 0.))[0]))
sns.ecdfplot(np.diff(np.where((plotdata["score"] > 0.))[0]))

# %%
np.corrcoef(lfc, scores)

# %%
np.corrcoef(np.log1p(bincounts2), scores)

# %%
w = [100000+33250,100000+33400]

# %%
plt.plot(scores[w[0]:w[1]])

# %%
bincounts = np.bincount(coordinates_oi, minlength=fragments.regions.window_width)[w[0]:w[1]]
plt.bar(np.arange(len(bincounts)), bincounts)

# %% [markdown]
# ## Data-driven PFM

# %%
fragments_oi = fragments.mapping[:, 1] == region_ix
coordinates_oi = fragments.coordinates[fragments_oi].flatten() - fragments.regions.window[0]
padding = 14
coordinates_oi = coordinates_oi[(coordinates_oi >= padding) & (coordinates_oi < fragments.regions.window_width - padding)]

# %%
onehot = region_onehots[gene_oi]
pfm = torch.stack([
    onehot[coordinates_oi+k].sum(0) for k in range(-padding, padding+1)
])

# %%
pwm = np.log2(pfm/pfm.sum(1)[:, None] / onehot.mean(0))
pwm_torch = pwm.float().T

# %%
sns.heatmap(pwm)

# %%
scores_original, positions, strand = chd.data.motifscan.motifscan.scan(
    region_onehots[list(region_onehots.keys())[0]].T[None, :, :], pwm_torch, cutoff = -99999999999.
)

padding = 15
fragments_oi = fragments.mapping[:, 1] == region_ix
coordinates_oi = fragments.coordinates[fragments_oi].flatten() - fragments.regions.window[0]
coordinates_oi = coordinates_oi[(coordinates_oi >= padding) & (coordinates_oi < fragments.regions.window_width - padding)]
bincounts = np.bincount(coordinates_oi, minlength = fragments.regions.window_width)

basepad1 = (fragments.regions.region_width - (scores_original.shape[0] // 2)) // 2
basepad2 = (fragments.regions.region_width - (scores_original.shape[0] // 2)) // 2

for l in range(-10, 10):
    scores = np.pad(scores_original.reshape(2, -1), ((0, 0), (basepad1-l, basepad2+l)), "constant", constant_values = 0).max(0)
    
    print(l, np.corrcoef(scores, np.log1p(bincounts))[0, 1])

# %%
l = 0
scores = np.pad(scores_original.reshape(2, -1), ((0, 0), (basepad1-l, basepad2+l)), "constant", constant_values = 0).max(0)

# %%
plotdata = pd.DataFrame({
    "bincount":bincounts,
    "score":scores,
    "lfc":lfc
})
plotdata["abslfc"] = np.abs(plotdata["lfc"])
plotdata["log1p_bincount"] = np.log1p(plotdata["bincount"])

# %%
fig = polyptich.grid.Figure(polyptich.grid.Grid())

panel, ax = fig.main.add_right(polyptich.grid.Panel((2, 2)))
ax.scatter(plotdata["log1p_bincount"], plotdata["score"], s = 1)
ax.set_xlabel("log1p(bincounts)")
ax.set_ylabel("scores")
cor = np.corrcoef(plotdata["log1p_bincount"], plotdata["score"])[0, 1]
ax.annotate(f"cor = {cor:.2f}", (0.95, 0.05), xycoords = "axes fraction", ha = "right", va = "bottom")
ax.set_ylim(-15, 15)

plotdata_oi = plotdata#[plotdata["log1p_bincount"] > 0.]

panel, ax = fig.main.add_right(polyptich.grid.Panel((2, 2)))
ax.scatter(plotdata_oi["lfc"], plotdata_oi["score"], s = 1)
ax.set_xlabel("lfc")
ax.set_ylabel("scores")
cor = np.corrcoef(plotdata_oi["lfc"], plotdata_oi["score"])[0, 1]
ax.annotate(f"cor = {cor:.2f}", (0.95, 0.05), xycoords = "axes fraction", ha = "right", va = "bottom")
ax.set_ylabel("")
ax.set_ylim(-15, 15)

panel, ax = fig.main.add_right(polyptich.grid.Panel((2, 2)))
ax.scatter(plotdata_oi["abslfc"], plotdata_oi["score"], s = 1)
ax.set_xlabel("abs(lfc)")
ax.set_ylabel("scores")
cor = np.corrcoef(plotdata_oi["abslfc"], plotdata_oi["score"])[0, 1]
ax.annotate(f"cor = {cor:.2f}", (0.95, 0.05), xycoords = "axes fraction", ha = "right", va = "bottom")
ax.set_ylabel("")
ax.set_ylim(-15, 15)

plotdata_oi = plotdata[plotdata["log1p_bincount"] > 0.]

panel, ax = fig.main.add_right(polyptich.grid.Panel((2, 2)))
ax.scatter(plotdata_oi["lfc"], plotdata_oi["score"], s = 1)
ax.set_xlabel("lfc")
ax.set_ylabel("scores")
cor = np.corrcoef(plotdata_oi["lfc"], plotdata_oi["score"])[0, 1]
ax.annotate(f"cor = {cor:.2f}", (0.95, 0.05), xycoords = "axes fraction", ha = "right", va = "bottom")
ax.set_ylabel("")
ax.set_ylim(-15, 15)

panel, ax = fig.main.add_right(polyptich.grid.Panel((2, 2)))
ax.scatter(plotdata_oi["abslfc"], plotdata_oi["score"], s = 1)
ax.set_xlabel("abs(lfc)")
ax.set_ylabel("scores")
cor = np.corrcoef(plotdata_oi["abslfc"], plotdata_oi["score"])[0, 1]
ax.annotate(f"cor = {cor:.2f}", (0.95, 0.05), xycoords = "axes fraction", ha = "right", va = "bottom")
ax.set_ylabel("")
ax.set_ylim(-15, 15)

fig.plot()

# %%
sns.ecdfplot(np.diff(np.where((plotdata["score"] > 1.))[0]))

# %%
np.where((plotdata["score"] > 1.))[0]

# %%
sns.ecdfplot(np.diff(np.where((plotdata["score"] > 1.))[0]))

# %%

# %%
# fig, ax = plt.subplots()
# ax.scatter(np.log1p(bincounts), scores, s = 1)
# ax.set_xlabel("log1p(bincounts)")
# ax.set_ylabel("scores")

# %%
np.corrcoef(np.log1p(bincounts2), scores)

# %%
np.corrcoef(lfc, scores)

# %%
np.corrcoef(np.abs(lfc), scores)

# %% [markdown]
# -----

# %%
deltacor = regionmultiwindow.interpolation["deltacor"].sel_xr(gene_oi).to_pandas().iloc[:-1]

# %%

# %%
pwm_torch

# %%
regionmultiwindow
