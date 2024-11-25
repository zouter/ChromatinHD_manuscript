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
import polyptich as pp
pp.setup_ipython()

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns
sns.set_style('ticks')

import pickle

import tqdm.auto as tqdm

import pathlib

import polars as pl

# %%
import peakfreeatac as pfa

# %%
folder_cons = pfa.get_output() / "data" / "cons" / "hs" / "gerp"
folder_cons.mkdir(exist_ok = True, parents=True)

# %% [markdown]
# http://hgdownload.soe.ucsc.edu/goldenPath/hg19/phastCons100way/

# %%
# !curl --location https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons100way/hg38.phastCons100way.bw > {folder_cons}/hg38.phastCons100way.bw

# %%
# !pip install pyBigWig

# %%
import pyBigWig
bw = pyBigWig.open(str(folder_cons/"hg38.phastCons100way.bw"))

# %%

# %%
# !wc -l {folder_qtl}/full.tsv

# %% [markdown]
# ## Create 

# %%
folder_root = pfa.get_output()
folder_data = folder_root / "data"

# dataset_name = "lymphoma"; organism = "hs"
dataset_name = "pbmc10k"; organism = "hs"
# dataset_name = "alzheimer"
dataset_name = "brain"; organism = "hs"

folder_data_preproc = folder_data / dataset_name

transcriptome = pfa.data.Transcriptome(folder_data_preproc / "transcriptome")

# %%
motiftrack_name = "conservation"

# %%
promoter_name, window = "10k10k", np.array([-10000, 10000])
# promoter_name, window = "1k1k", np.array([-1000, 1000])
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)

# %%
promoter = promoters.loc[transcriptome.gene_id("AAK1")]

# %%
plt.plot(bw.values(promoter["chr"], promoter["start"], promoter["end"]))

# %% [markdown]
# ### Link QTLs to SNP location

# %%
promoter

# %%
scores = []

for gene_ix, promoter in enumerate(promoters.itertuples()):
    conservation = np.array(bw.values(promoter.chr, promoter.start, promoter.end))[:, None]
    
    if promoter.strand == -1:
        conservation = conservation[::-1]
    
    scores.append(conservation)
scores = np.vstack(scores)

# %% [markdown]
# ### Save

# %%
import peakfreeatac as pfa

# %%
motiftrack = pfa.data.Motiftrack(pfa.get_output() / "motiftracks" / dataset_name / promoter_name / motiftrack_name)

# %%
motiftrack.scores = scores

# %%
motifs = pd.DataFrame([
    ["conservation"]
], columns = ["motif"]).set_index("motif")

# %%
pickle.dump(motifs, open(motiftrack.path / "motifs.pkl", "wb"))

# %%
# !ls -lh {motiftrack.path}

# %%
