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

import pybedtools

import tqdm.auto as tqdm

# %%
import chromatinhd as chd
import tempfile
import requests

# %%
bed_folder = chd.get_output() / "bed" / "gm1282_tf_chipseq"
bed_folder.mkdir(exist_ok = True, parents = True)

filtered_bed_folder = chd.get_output() / "bed" / "gm1282_tf_chipseq_filtered"
filtered_bed_folder.mkdir(exist_ok = True, parents = True)

files = pd.read_csv(bed_folder / "files.csv", index_col = 0)
# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

# transcriptome
dataset_name = "pbmc10k"
folder_dataset = chd.get_output() / "datasets"/ dataset_name
regions = chd.data.Regions(folder_dataset / "regions" / "100k100k")
promoters = regions.coordinates

# %%
def center_peaks(peaks, promoter):
    if peaks.shape[0] == 0:
        peaks = pd.DataFrame(columns=["start", "end", "method"])
    else:
        peaks[["start", "end"]] = [
            [
                (peak["start"] - promoter["tss"]) * promoter["strand"],
                (peak["end"] - promoter["tss"]) * promoter["strand"],
            ][:: int(promoter["strand"])]
            for _, peak in peaks.iterrows()
        ]
    return peaks

# %%
sites = []
for gene in tqdm.tqdm(promoters.index):
    promoter = promoters.loc[gene]
    for file_accession, file in files.iterrows():
        filtered_file = filtered_bed_folder / f"{file.accession}_{gene}.bed"

        if filtered_file.exists() and filtered_file.stat().st_size > 0:
            # read bed file using pandas
            bed = pd.read_table(filtered_file, names = ["chrom", "start", "end", "strand", "score", "signal", "pvalue", "qvalue", "peak", "."])
            # bed = pybedtools.BedTool(filtered_file).to_dataframe()
            if len(bed):
                bed = bed[["chrom", "start", "end"]]
                bed = center_peaks(bed, promoter)

                bed["gene"] = gene
                bed["file_accession"] = file.accession
                sites.append(bed)

sites = pd.concat(sites)

# %%
sites["start"].plot(kind = "hist", bins = 100)
# %%
len(sites)
# %%
sites["file_accession"].value_counts()
# %%
sites.groupby("file_accession").size().to_frame("n").join(files.set_index("accession")).sort_values("n", ascending = False).head(10)
# %%
sites.to_csv(filtered_bed_folder / "sites.csv")
# %%
