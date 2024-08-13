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
bed_folder = chd.get_output() / "bed" / "pu_mcp_chipseq"
bed_folder.mkdir(exist_ok = True, parents = True)

url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE128nnn/GSE128834/suppl/GSE128834_RAW.tar"

# %%
!wget {url} -O {bed_folder / "GSE128834_RAW.tar"}

# %%
!tar -xvf {bed_folder / "GSE128834_RAW.tar"} -C {bed_folder}

# %%
obj = requests.get(url)

# %%
import json
import io

# %%
files = pd.read_table(io.StringIO(obj.content.decode("utf-8")))
files.columns = ["_".join(col.lower().split(" ")) for col in files.columns]
files = files.set_index("file_accession")

# %%
files.head()
files["file_format"].value_counts()
files["output_type"].value_counts()

# %%
files = files.loc[files["output_type"] == "fold change over control"]
files = files.loc[files["file_assembly"] == "GRCh38"]
files = files.groupby("experiment_target").first().reset_index()
files["experiment_target"].value_counts()

# %%
files = files.loc[files["experiment_target"].str.contains("POU2F2") | files["experiment_target"].str.contains("TCF3") | files["experiment_target"].str.contains("SPI1") | files["experiment_target"].str.contains("RELA") | files["experiment_target"].str.contains("EBF1")| files["experiment_target"].str.contains("IRF4")| files["experiment_target"].str.contains("PAX5")| files["experiment_target"].str.contains("ZEB1")| files["experiment_target"].str.contains("RELB")| files["experiment_target"].str.contains("IRF1")]
files["experiment_target"].value_counts()

# %%
files["filename"] = files["file_download_url"].str.split("/").str[-1]
files["accession"] = files["file_download_url"].str.split("/").str[-1].str.split(".").str[0]

# %%
files.to_csv(bed_folder / "files.csv")

# %% [markdown]
# ### Download files

# %%
encode_folder = chd.get_output() / "data" / "encode"
encode_folder_relative = encode_folder.relative_to(chd.get_git_root())

# %%
# !ln -s ~/NAS2/wsaelens/projects/chromatinhd/chromatinhd_manuscript/{encode_folder_relative} {encode_folder}


# %%
import urllib

# %%
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

# %%
for _, file in files.iterrows():
    if not (bed_folder / file["filename"]).exists():
        print(file["filename"])
        urllib.request.urlretrieve(file["file_download_url"], bed_folder / file["filename"], )



# %%





