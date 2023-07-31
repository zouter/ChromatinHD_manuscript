# %%
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns
sns.set_style('ticks')
%config InlineBackend.figure_format='retina'

import pickle

import scanpy as sc

import pathlib

import torch

import tqdm.auto as tqdm

# %%
import chromatinhd as chd
import tempfile
import requests

# %%
url = "https://www.encodeproject.org/metadata/?assay_title=WGBS&assay_title=Mint-ChIP-seq&biosample_ontology.term_name=naive+thymus-derived+CD8-positive%2C+alpha-beta+T+cell&biosample_ontology.term_name=naive+thymus-derived+CD4-positive%2C+alpha-beta+T+cell&biosample_ontology.term_name=naive+B+cell&biosample_ontology.term_name=B+cell&biosample_ontology.term_name=CD14-positive+monocyte&biosample_ontology.term_name=natural+killer+cell&biosample_ontology.term_name=immature+natural+killer+cell&biosample_ontology.term_name=dendritic+cell&assay_title=Histone+ChIP-seq&control_type%21=%2A&files.file_type=bigWig&type=Experiment&files.analyses.status=released&files.preferred_default=true"

# %%
obj = requests.get(url)

# %%
import json

# %%
import io

# %%
files = pd.read_table(io.StringIO(obj.content.decode("utf-8")))

# %%
files.to_csv(bw_folder / "files.csv")

# %%
assert biosamples_oi["Biosample term name"].isin(files["Biosample term name"]).all()
assert files["Biosample term name"].isin(biosamples_oi["Biosample term name"]).all()

# %% [markdown]
# ### Download files

# %%
encode_folder = chd.get_output() / "data" / "encode"
encode_folder_relative = encode_folder.relative_to(chd.get_git_root())

# %%
bw_folder = encode_folder / "immune"
bw_folder.mkdir(exist_ok = True, parents = True)

# %%
!ln -s ~/NAS2/wsaelens/projects/chromatinhd/chromatinhd_manuscript/{encode_folder_relative} {encode_folder}

# %%
files["filename"] = files["File download URL"].str.split("/").str[-1]

# %%
import urllib

# %%
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

# %%
for _, file in files.iterrows():
    if not (bw_folder / file["filename"]).exists():
        print(file["filename"])
        urllib.request.urlretrieve(file["File download URL"], bw_folder / file["filename"], )

# %%



