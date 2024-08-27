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
def process_gene(gene):
    print(gene)
    promoter = promoters.loc[gene]

    promoter_start = promoter["start"]
    promoter_end = promoter["end"]
    promoter_chrom = promoter["chrom"]

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
