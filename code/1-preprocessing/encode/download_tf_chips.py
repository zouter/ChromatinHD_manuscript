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
bed_folder = chd.get_output() / "bed" / "gm1282_tf_chipseq"
bed_folder.mkdir(exist_ok = True, parents = True)

# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

# %%
url = "https://www.encodeproject.org/metadata/?type=Experiment&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&assay_title=TF+ChIP-seq&status=released&biosample_ontology.term_name=GM12878&target.label!=CTCF&target.label!=EP300&target.label!=POLR2A&assembly=GRCh38&control_type!=*&target.label!=POLR2AphosphoS2&target.label!=POLR2AphosphoS5"
# url = "https://www.encodeproject.org/search/?type=Experiment&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&assay_title=TF+ChIP-seq&status=released&biosample_ontology.term_name=GM12878&target.label!=CTCF&target.label!=EP300&target.label!=POLR2A&assembly=GRCh38&control_type!=*&target.label!=POLR2AphosphoS2&target.label!=POLR2AphosphoS5"

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
files = files.loc[files["file_format"] == "bed narrowPeak"]
files = files.loc[files["output_type"].isin(["optimal IDR thresholded peaks", "conservative IDR thresholded peaks"])]
files = files.loc[files["file_assembly"] == "GRCh38"]
files = files.groupby("experiment_target").first().reset_index()
files["experiment_target"].value_counts()

# %%
files["filename"] = files["file_download_url"].str.split("/").str[-1]
files["accession"] = files["file_download_url"].str.split("/").str[-1].str.split(".").str[0]

# %%
files.to_csv(bed_folder / "files.csv")

# %%
files_oi = files[["experiment_target", "accession"]]
manuscript.store_supplementary_table(files_oi, f"chipseqs_tfs")

# %% [markdown]
# ### Download files

# %%
encode_folder = chd.get_output() / "data" / "encode"
encode_folder_relative = encode_folder.relative_to(chd.get_git_root())

# %%
bed_folder = chd.get_output() / "bed" / "gm1282_tf_chipseq"
bed_folder.mkdir(exist_ok = True, parents = True)

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
files["filename_sorted"] = files["filename"].str.split(".").str[0] + ".sorted.bed"
for _, file in files.iterrows():
    if not (bed_folder / file["filename_sorted"]).exists():
        print(file["filename_sorted"])
        !bedtools sort -i {bed_folder / file["filename"]} > {bed_folder / file["filename_sorted"]}

# %%
import pybedtools

# %%
!wget https://www.encodeproject.org/files/GRCh38_EBV.chrom.sizes/@@download/GRCh38_EBV.chrom.sizes.tsv -O {chd.get_output()}/chromosome.sizes
chromosome_sizes = chd.get_output() / "chromosome.sizes"

final_files = {}
for file_accession, file in files.iterrows():
    final_files[file_accession] = pybedtools.BedTool(bed_folder / file["filename_sorted"]).slop(b = 100, g = chromosome_sizes)


# %%
overlaps = []
for (file_accession_a, file_a), (file_accession_b, file_b) in tqdm.tqdm(itertools.combinations(files.iterrows(), 2)):
    if not (("SPI" in file_a["experiment_target"]) | ("SPI" in file_b["experiment_target"])):
        continue
    bed_a = final_files[file_accession_a]
    bed_b = final_files[file_accession_b]
    overlap_a = bed_a.intersect(bed_b, u = True).count()
    overlap_b = bed_b.intersect(bed_a, u = True).count()

    overlaps.append({
        "file_a": file_accession_a,
        "file_b": file_accession_b,
        "overlap_ab": overlap_a,
        "overlap_ba": overlap_b,
        "n_a": bed_a.count(),
        "n_b": bed_b.count(),
        "jaccard": overlap_a / (bed_a.count() + bed_b.count() - overlap_a)
    })

# %%
overlaps = pd.DataFrame(overlaps)

# %%
# %%
overlaps.sort_values("jaccard")
# %%
overlaps["target_a"] = files.loc[overlaps["file_a"]]["experiment_target"].str.split("-").str[0].values
overlaps["target_b"] = files.loc[overlaps["file_b"]]["experiment_target"].str.split("-").str[0].values

# %%
sns.heatmap(overlaps.set_index(["target_a", "target_b"])["jaccard"].unstack())

# %%
# %%
overlaps.sort_values("jaccard")

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
final_files = {}
for file_accession, file in files.iterrows():
    final_files[file_accession] = pybedtools.BedTool(bed_folder / file["filename_sorted"])

# %%
for gene, promoter in tqdm.tqdm(promoters.iterrows(), total = len(promoters), desc = "Genes"):
    promoter_start = promoter["start"]
    promoter_end = promoter["end"]
    promoter_chrom = promoter["chr"]

    for file_accession, file in tqdm.tqdm(files.iterrows(), total = len(files), desc = gene, leave = False):
        filtered_file = bed_folder / f"{file_accession}_{gene}.bed"

        if not filtered_file.exists():
            bed = final_files[file_accession]
            bed = bed.filter(lambda x: x.chrom == promoter_chrom)
            bed = bed.filter(lambda x: x.start >= promoter_start)
            bed = bed.filter(lambda x: x.end <= promoter_end)
            bed = bed.sort()

            bed.saveas(bed_folder / f"{file_accession}_{gene}.bed")
# %%
