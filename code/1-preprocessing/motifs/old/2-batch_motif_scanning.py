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
# %config InlineBackend.figure_format='retina'

import pickle

import scanpy as sc
import pathlib

import torch_scatter
import torch
import math

import tqdm.auto as tqdm

device = "cuda:0"

# %%
import chromatinhd as chd

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

# dataset_name = "lymphoma"
# organism = "hs"
dataset_name = "pbmc10k"
organism = "hs"
# dataset_name = "e18brain"; organism = "mm"
# dataset_name = "alzheimer"; organism = "mm"
# dataset_name = "brain"; organism = "hs"
# dataset_name = "pbmc10k_gran"; organism = "hs"
# dataset_name = "GSE198467_H3K27ac"; organism = "mm"; genome = "mm10"
# dataset_name = "GSE198467_single_modality_H3K27me3"; organism = "mm"; genome = "mm10"

# dataset_name = "FLI1_7"
# dataset_name = "PAX2_7"
# dataset_name = "NHLH1_7"
# dataset_name = "CDX2_7"
# dataset_name = "CDX1_7"
# dataset_name = "MSGN1_7"
# dataset_name = "KLF4_7"
# dataset_name = "KLF5_7"
# dataset_name = "PTF1A_4"
# dataset_name = "morf_20"; organism = "hs"

folder_data_preproc = folder_data / dataset_name


# %%
folder_motifs = chd.get_output() / "data" / "motifs" / organism / "hocomoco"
folder_motifs.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Process promoter sequences

# %%
# promoter_name, window = "100k100k", np.array([-100000, 100000])
promoter_name, window = "10k10k", np.array([-10000, 10000])
# promoter_name, window = "1k1k", np.array([-1000, 1000])
promoters = pd.read_csv(
    folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
)

# %% [markdown]
# ## Motif scanning in promoters

# %%
onehot_promoters = pickle.load(
    (folder_data_preproc / ("onehot_promoters_" + promoter_name + ".pkl")).open("rb")
)

# %%
pwms = pickle.load((folder_motifs / "pwms.pkl").open("rb"))
motifs = pd.read_pickle(folder_motifs / "motifs.pkl")

# motifs_oi = motifs.loc[motifs["gene_label"].isin(["TCF7", "GATA3", "IRF4"])]
motifs_oi = motifs.iloc[:1]
# motifs_oi = motifs.iloc[:20]
# motifs_oi = motifs.loc[
#     motifs["gene_label"].isin(
#         [
#             "SPI1",
#         ]
#     )
# ]
motifs_oi = motifs

# %%
nucleotides = pd.DataFrame({"nucleotide": np.arange(4), "label": ["A", "C", "G", "T"]})
nucleotides["color"] = sns.color_palette(n_colors=4)

# %%
# motif_oi = "ZN250_HUMAN.H11MO.0.C"
# motif_oi = "ZN250_HUMAN.H11MO.0.C"
motif_oi = "SALL4_HUMAN.H11MO.0.B"
# motif_oi = "CEBPA_MOUSE.H11MO.0.A"
print(motif_oi)
fig, ax = plt.subplots()
pd.DataFrame(pwms[motif_oi].numpy()).plot(ax=ax)
ax.axhline(0, color="#333333")


# %%
def scan(onehot, pwm):
    n = onehot.shape[-2]
    k = pwm.shape[-2]

    # forward strand
    positive = torch.zeros(((*onehot.shape[:-2], n - k + 1)), device=onehot.device)
    for i in range(k):
        # to save memory we do the matrix multiplication once per motif position
        # this does not cause a significant slowdown
        x = torch.matmul(onehot, pwm[[i]].T)
        positive += x[..., i : n - k + i + 1, 0]
    del x

    # reverse (complement) strand
    onehot_comp = onehot[..., [3, 2, 1, 0]]
    pwm_rev = pwm.flip(0)
    negative = torch.zeros(((*onehot.shape[:-2], n - k + 1)), device=onehot.device)
    for i in range(k):
        x = torch.matmul(onehot_comp, pwm_rev[[i]].T)
        negative += x[..., i : n - k + i + 1, 0]
    del x

    # return maximum score across forward or reverse strands
    return torch.maximum(positive, negative)


# unit test
onehot = torch.tensor(np.eye(4, dtype=np.float32)[np.array([0, 1, 2, 3, 3, 2, 1, 0])])[
    None, ...
]
pwm = torch.tensor([[1, 0.0, 0.0, 0.0], [0.0, 1, 0.0, 0.0]])
motifscore = scan(onehot, pwm)
assert motifscore.shape[1] == 8 - 2 + 1
assert (motifscore == torch.tensor([[2.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0]])).all()

# %%
# theoretical memory consumption when put on cuda
str(np.prod(onehot_promoters.shape) * 32 / 8 / 1024 / 1024 / 1024) + " GiB"

# %% [markdown]
# Running this on GPU generally gives a 10x time improvement (40 minutes to 3 minutes)

# %%
cutoff_col = "cutoff_0001"
# cutoff_col = "cutoff_001"

# %%
device = "cuda"
# device = "cpu"

# %% tags=[]
position_ixs = []
motif_ixs = []
scores = []

batch = 20000

for batch_ix in tqdm.tqdm(range(math.ceil(onehot_promoters.shape[1] / batch))):
    onehot_promoters_oi = onehot_promoters[
        :, batch_ix * batch : ((batch_ix + 1) * batch + 100)
    ].to(device)
    for motif_ix, motif in enumerate(tqdm.tqdm(motifs_oi.index)):
        cutoff = motifs_oi.loc[motif, cutoff_col]
        pwm = pwms[motif].to(onehot_promoters_oi.device)
        score = scan(onehot_promoters_oi, pwm)
        pad = onehot_promoters_oi.shape[-2] - score.shape[-1]
        pad_left = math.ceil(pad / 2)
        pad_right = math.floor(pad / 2)
        shape_left = (*score.shape[:-1], pad_left)
        shape_right = (*score.shape[:-1], pad_right)
        score = torch.cat(
            [
                torch.zeros(shape_left, device=score.device),
                score,
                torch.zeros(shape_right, device=score.device),
            ],
            dim=-1,
        )

        # add to sparse container
        # position in this case refers to gene x position (based on promoter window)
        position_ix = torch.where(score.flatten() > cutoff)[0].cpu().numpy()

        local_position_ix = position_ix % score.shape[1]
        gene_ix = position_ix // score.shape[1]

        position_ixs.extend(
            local_position_ix + (batch_ix * batch) + gene_ix * onehot_promoters.shape[1]
        )
        motif_ixs.extend(np.repeat(motif_ix, len(position_ix)))
        scores.extend(score.flatten().cpu().numpy()[position_ix])

    onehot_promoters_oi = onehot_promoters_oi.to("cpu")
    del onehot_promoters_oi

# %%
import scipy.sparse

# convert to csr, but using coo as input
motifscores = scipy.sparse.csr_matrix(
    (scores, (position_ixs, motif_ixs)),
    shape=(np.prod(onehot_promoters.shape[:2]), motifs_oi.shape[0]),
)

# %%
local_position_ixs = np.array(position_ixs) % onehot_promoters.shape[1]

# %%
position_ixs_oi = local_position_ixs[
    (local_position_ixs > (onehot_promoters.shape[1] // 2) - 10000)
    & (local_position_ixs < (onehot_promoters.shape[1] // 2) + 10000)
]
plt.hist(position_ixs_oi, bins=20)

# %% [markdown]
# Assess whether motif scanning worked correctly

# %%
# motif_ix = 0
# motif_ix = motifs_oi.index.tolist().index("ZN250_HUMAN.H11MO.0.C")
motif_ix = motifs_oi.index.tolist().index(
    motifs_oi.index[motifs_oi.index.str.contains("TCF7")][0]
)

# %%
gene_ix = 20
# gene_ix = promoters.index.tolist().index("ENSG00000115977")
pwm = pwms[motifs_oi.iloc[motif_ix].name]

# %%
chr, start, end, strand = promoters.iloc[gene_ix][["chr", "start", "end", "strand"]]

# %%
window_length = window[1] - window[0]

# %%
max_pos = motifscores[
    (gene_ix * window_length) : ((gene_ix + 1) * window_length), motif_ix
].argmax()
max_score = motifscores[
    (gene_ix * window_length) : ((gene_ix + 1) * window_length), motif_ix
].max()

# %%
# score = scan(onehot_promoters, pwm)
# max = score[gene_ix].max(0)
# max

# %%
# maximum possible score
pwm.max(1)[0].sum()

# %%
local_start = max_pos - math.floor(pwm.shape[0] / 2)
local_end = max_pos + math.ceil(pwm.shape[0] / 2)

# %%
# check score using a manual multiplication
forward_score = (
    onehot_promoters[gene_ix, local_start:local_end].numpy() * pwm.numpy()
).sum()
reverse_score = (
    onehot_promoters[gene_ix, local_start:local_end].numpy()[::-1, [3, 2, 1, 0]]
    * pwm.numpy()
).sum()

assert np.isclose(np.max([forward_score, reverse_score]), max_score)
forward_score, reverse_score

# %%
locus_start = start + max_pos - window[0]
locus_end = start + max_pos - window[0] + pwm.shape[0]

# %%
onehot = onehot_promoters[gene_ix, local_start:local_end]

# %%
fig, (ax_score, ax_onehot, ax_pwm, ax_onehotrev, ax_scorerev) = plt.subplots(
    5, 1, figsize=(3, 4), sharex=True
)

ntscores = pwm.flatten()[onehot.flatten().to(bool)]
ax_score.fill_between(np.arange(onehot.shape[0]), ntscores, color="#55555533")
ax_score.scatter(
    np.arange(onehot.shape[0]),
    ntscores,
    c=np.array(sns.color_palette(n_colors=4))[onehot.argmax(1)],
)
ax_score.set_ylabel("Forward scores", rotation=0, ha="right", va="center")

pd.DataFrame(onehot.numpy()).plot(ax=ax_onehot, legend=False)
ax_onehot.set_ylabel("Forward sequence", rotation=0, ha="right", va="center")

pd.DataFrame(pwm.numpy()).plot(ax=ax_pwm, legend=False)
ax_pwm.set_ylabel("PWM", rotation=0, ha="right", va="center")

pd.DataFrame(onehot.numpy()[::-1, [3, 2, 1, 0]]).plot(ax=ax_onehotrev, legend=False)
ax_onehotrev.set_ylabel("Reverse sequence", rotation=0, ha="right", va="center")

onehot_rev = onehot.numpy()[::-1, [3, 2, 1, 0]]
ntscores = pwm.flatten()[onehot_rev.flatten().astype(bool)]
ax_scorerev.fill_between(np.arange(onehot.shape[0]), ntscores, color="#55555533")
ax_scorerev.scatter(
    np.arange(onehot.shape[0]),
    ntscores,
    c=np.array(sns.color_palette(n_colors=4))[onehot_rev.argmax(1)],
)
ax_scorerev.set_ylabel("Reverse scores", rotation=0, ha="right", va="center")

# %% [markdown]
# ### Create motifscan

# %%
motifscan_name = cutoff_col

# %%
import chromatinhd as chd

# %%
motifscan = chd.data.Motifscan(
    chd.get_output() / "motifscans" / dataset_name / promoter_name / motifscan_name
)

# %% [markdown]
# ### Save motif indices

# %%
motifscan.indices = motifscores.indices
motifscan.indptr = motifscores.indptr
motifscan.data = motifscores.data
motifscan.shape = motifscores.shape

# %% [markdown] tags=[]
# ### Save motifs (with gene info)

# %%
biomart_dataset_name = (
    "mmusculus_gene_ensembl" if organism == "mm" else "hsapiens_gene_ensembl"
)

# %%
query = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query  virtualSchemaName = "default" formatter = "TSV" header = "1" uniqueRows = "0" count = "" datasetConfigVersion = "0.6" >

    <Dataset name = "{biomart_dataset_name}" interface = "default" >
        <Attribute name = "ensembl_gene_id" />
        <Attribute name = "entrezgene_id" />
    </Dataset>
</Query>"""
url = "http://www.ensembl.org/biomart/martservice?query=" + query.replace(
    "\t", ""
).replace("\n", "")
from io import StringIO
import requests

session = requests.Session()
session.headers.update({"User-Agent": "Custom user agent"})
r = session.get(url)
result = pd.read_table(StringIO(r.content.decode("utf-8")))

# %%
result = result.rename(
    columns={
        "NCBI gene (formerly Entrezgene) ID": "EntrezGene",
        "Gene stable ID": "gene",
    }
)
result = result.dropna()
result = result.reset_index()
result["EntrezGene"] = result["EntrezGene"].astype(int).astype(str)
result = result.groupby("EntrezGene").first()

# %%
motifs_oi["gene"] = result.reindex(motifs_oi["EntrezGene"])["gene"].tolist()

# %%
pd.isnull(motifs_oi["gene"]).sum()

# %%
pickle.dump(motifs_oi, open(motifscan.path / "motifs.pkl", "wb"))

# %%
# !ls -lh {motifscan.path}

# %%
motifscan.n_motifs = len(motifs_oi)
