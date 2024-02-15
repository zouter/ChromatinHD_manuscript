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
bed_folder = chd.get_output() / "bed" / "wtc11_tf_chipseq_bw"
url = "https://www.encodeproject.org/metadata/?type=Experiment&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&assay_title=TF+ChIP-seq&status=released&biosample_ontology.term_name=WTC11&target.label!=CTCF&target.label!=EP300&target.label!=POLR2A&assembly=GRCh38&control_type!=*&target.label!=POLR2AphosphoS2&target.label!=POLR2AphosphoS5"
targets = [
    "ERG"
]

# bed_folder = chd.get_output() / "bed" / "gm1282_tf_chipseq_bw"
# url = "https://www.encodeproject.org/metadata/?type=Experiment&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&assay_title=TF+ChIP-seq&status=released&biosample_ontology.term_name=GM12878&target.label!=CTCF&target.label!=EP300&target.label!=POLR2A&assembly=GRCh38&control_type!=*&target.label!=POLR2AphosphoS2&target.label!=POLR2AphosphoS5"
# targets = [
#     # "POU2F2", "ETV6", "IRF4",
#     "POU2F2", "ETV6" ,"TCF3" ,"SPI1" ,"RELA" ,"EBF1" ,"IRF4" ,"PAX5" ,"ZEB1" ,"RELB" ,"IRF1" ,"RUNX3" ,"TBX21" ,"TCF7", "IRF7",
# ]

bed_folder = chd.get_output() / "bed" / "k562_tf_chipseq_bw"
url = "https://www.encodeproject.org/metadata/?type=Experiment&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&assay_title=TF+ChIP-seq&status=released&biosample_ontology.term_name=K562&target.label!=CTCF&target.label!=EP300&target.label!=POLR2A&assembly=GRCh38&control_type!=*&target.label!=POLR2AphosphoS2&target.label!=POLR2AphosphoS5"
targets = [
    "IRF9",
    "SP2",
    "GATA1", 
    "GATA2", 
    "TAL1", 
    "KLF1",
    "NFE2",
    ]


# bed_folder = chd.get_output() / "bed" / "erythroblast_tf_chipseq_bw"
# url = "https://www.encodeproject.org/metadata/?type=Experiment&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&assay_title=TF+ChIP-seq&status=released&biosample_ontology.term_name=erythroblast&target.label!=CTCF&target.label!=EP300&target.label!=POLR2A&assembly=GRCh38&control_type!=*&target.label!=POLR2AphosphoS2&target.label!=POLR2AphosphoS5"
# targets = ["GATA1", "GATA2"]

# bed_folder = chd.get_output() / "bed" / "hl60_tf_chipseq_bw"
# url = "https://www.encodeproject.org/metadata/?type=Experiment&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&assay_title=TF+ChIP-seq&status=released&biosample_ontology.term_name=HL-60&target.label!=CTCF&target.label!=EP300&target.label!=POLR2A&assembly=GRCh38&control_type!=*&target.label!=POLR2AphosphoS2&target.label!=POLR2AphosphoS5"
# targets = ["SPI1"]

# bed_folder = chd.get_output() / "bed" / "gm12891_tf_chipseq_bw"
# url = "https://www.encodeproject.org/metadata/?type=Experiment&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&assay_title=TF+ChIP-seq&status=released&biosample_ontology.term_name=GM12891&target.label!=CTCF&target.label!=EP300&target.label!=POLR2A&assembly=GRCh38&control_type!=*&target.label!=POLR2AphosphoS2&target.label!=POLR2AphosphoS5"
# targets = [
#     "POU2F2", "ETV6",
# ]


# bed_folder = chd.get_output() / "bed" / "gm12891_tf_chipseq_bw"
# url = "https://www.encodeproject.org/metadata/?type=Experiment&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&assay_title=TF+ChIP-seq&status=released&biosample_ontology.term_name=GM12891&target.label!=CTCF&target.label!=EP300&target.label!=POLR2A&assembly=GRCh38&control_type!=*&target.label!=POLR2AphosphoS2&target.label!=POLR2AphosphoS5"
# targets = [
#     "POU2F2", "ETV6",
# ]

if not bed_folder.exists():
    bed_folder.mkdir(exist_ok = True, parents = True)

# %%
obj = requests.get(url)

# %%
import json
import io

# %%
files = pd.read_table(io.StringIO(obj.content.decode("utf-8")))
files.columns = ["_".join(col.lower().split(" ")) for col in files.columns]
files = files.set_index("file_accession")

files.head()
files["file_format"].value_counts()
files["output_type"].value_counts()

# %%
files = files.loc[files["output_type"] == "fold change over control"]
files = files.loc[files["file_assembly"] == "GRCh38"]
# files = files.groupby("experiment_target").reset_index()
files["experiment_target"].value_counts()

files_oi = pd.Series(False, files.index)
for target in targets:
    files_oi[files["experiment_target"] == (target + "-human")] = True
files = files.loc[files_oi]
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
str(bed_folder / file["filename"])
# %%
