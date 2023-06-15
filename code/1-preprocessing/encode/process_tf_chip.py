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
# dataset_name = "lymphoma"
dataset_name = "pbmc10k"
# dataset_name = "pbmc10k_gran"
# dataset_name = "e18brain"
folder_data_preproc = folder_data / dataset_name

# fragments
promoter_name = "100k100k"
promoters = pd.read_csv(
    folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
)

# %%
def process_gene(gene):
    print(gene)
    promoter = promoters.loc[gene]

    promoter_start = promoter["start"]
    promoter_end = promoter["end"]
    promoter_chrom = promoter["chr"]

    for file_accession, file in files.iterrows():
        filtered_file = bed_folder / f"{file.accession}_{gene}.bed"

        if not filtered_file.exists():
            bed = pybedtools.BedTool(bed_folder / file["filename"])
            bed = bed.filter(lambda x: x.chrom == promoter_chrom)
            bed = bed.filter(lambda x: x.start >= promoter_start)
            bed = bed.filter(lambda x: x.end <= promoter_end)
            bed = bed.sort()

            bed.saveas(filtered_bed_folder / f"{file.accession}_{gene}.bed") 
# %%

if __name__ == "__main__":
    import multiprocessing as mp
    from functools import partial

    pool = mp.Pool(16)
    pool.map(process_gene, promoters.index.tolist())

# %%
