# %%
# %load_ext autoreload
# %autoreload 2

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

import torch

import tqdm.auto as tqdm

# %%
import chromatinhd as chd
import tempfile
import requests

# %%
bed_folder = chd.get_output() / "bed" / "gm1282_structural_chipseq"
bed_folder.mkdir(exist_ok=True, parents=True)

# %%
url = "https://www.encodeproject.org/metadata/?type=Experiment&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&assay_title=TF+ChIP-seq&status=released&biosample_ontology.term_name=GM12878&target.label!=EP300&assembly=GRCh38&control_type!=*&target.label!=POLR2AphosphoS2&target.label!=POLR2AphosphoS5"

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
files = files.loc[files["file_format"] == "bigWig"]
files = files.loc[files["output_type"] == "fold change over control"]
files = files.loc[files["file_assembly"] == "GRCh38"]
files = files.groupby("experiment_target").first().reset_index()
files["target"] = files["experiment_target"].str.split("-").str[0]

files = files.loc[
    files["target"].isin(
        [
            "YY1",
            "ZNF143",
            "RAD21",
            "CTCF",
            "POLR2A",
            "SPI1",
            "EBF1",
            "PAX5",
            "RUNX1",
            "IRF4",
            "IRF8",
        ]
    )
]
files["experiment_target"].value_counts()

# %%
files["filename"] = files["file_download_url"].str.split("/").str[-1]
files["accession"] = files["file_download_url"].str.split("/").str[-1].str.split(".").str[0]

# %%
files["target"].value_counts()

# %%
files = files.set_index("accession")

# %%
files.to_csv(bed_folder / "files.csv")

# %% [markdown]
# ### Download files

# %%
encode_folder = chd.get_output() / "data" / "encode"
encode_folder_relative = encode_folder.relative_to(chd.get_git_root())

# %%
bed_folder = chd.get_output() / "bed" / "gm1282_structural_chipseq"
bed_folder.mkdir(exist_ok=True, parents=True)

# %%
# !ln -s ~/NAS2/wsaelens/projects/chromatinhd/chromatinhd_manuscript/{encode_folder_relative} {encode_folder}

# %%
import urllib

# %%
opener = urllib.request.build_opener()
opener.addheaders = [("User-agent", "Mozilla/5.0")]
urllib.request.install_opener(opener)

# %%
for _, file in files.iterrows():
    if not (encode_folder / file["filename"]).exists():
        print(file["filename"])
        urllib.request.urlretrieve(
            file["file_download_url"],
            encode_folder / file["filename"],
        )

# %% [markdown]
# ## Filter files around genes

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

# transcriptome
# dataset_name = "lymphoma"
dataset_name = "pbmc10k"
# dataset_name = "pbmc10k_gran"
# dataset_name = "e18brain"
folder_data_preproc = folder_data / dataset_name

regions = chd.data.Regions(chd.get_output() / "datasets" / dataset_name / "regions" / "100k100k")
promoters = regions.coordinates

# %%
import pyBigWig

# %%
import mgzip
import pickle

for file_accession, file in files.iterrows():
    print("processing", file_accession)
    # if (bed_folder / f"values_{file_accession}.pickle").exists() and file_accession != "ENCFF129YCR":
    #     continue
    # if file_accession != "ENCFF129YCR":
    #     continue

    values = {}
    bw = pyBigWig.open(str(encode_folder / file["filename"]))
    chroms = bw.chroms()
    for gene, promoter in tqdm.tqdm(promoters.iterrows(), total=len(promoters), desc="Genes"):
        promoter_start = promoter["start"]
        promoter_end = promoter["end"]
        promoter_chrom = promoter["chrom"]

        minstart = 0
        maxend = chroms[promoter["chrom"]]

        if promoter_start < minstart:
            promoter_start = minstart
        if promoter_end > maxend:
            promoter_end = maxend

        val = bw.values(promoter_chrom, promoter_start, promoter_end, numpy=True)
        if len(val) < promoter.end - promoter.start:
            pad_start = promoter_start - promoter.start
            pad_end = promoter.end - promoter_end
            val = np.pad(
                val,
                (pad_start, pad_end),
            )
        if promoter["strand"] == -1:
            val = val[::-1]
        values[(file_accession, gene)] = val

    with mgzip.open(str(bed_folder / f"values_{file_accession}.pickle"), "wb") as outfile:
        pickle.dump(values, outfile)

# %%


# %%
# !ls -lh {bed_folder}
# %%
