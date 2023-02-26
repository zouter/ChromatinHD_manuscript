# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns
sns.set_style('ticks')
# %config InlineBackend.figure_format='retina'

import pickle

import scanpy as sc

import pathlib

import tqdm.auto as tqdm

import chromatinhd as chd
import tempfile
import requests

# %% [markdown]
# ### Prepare folder

# %% tags=[]
data_folder = chd.get_output() / "data" / "eqtl" / "onek1k"
raw_data_folder = data_folder / "raw"
nas2_raw_data_folder = "/home/wsaelens/NAS2/wsaelens/projects/chromatinhd/chromatinhd_manuscript/output/" / raw_data_folder.relative_to(chd.get_output())

nas2_raw_data_folder.mkdir(exist_ok=True, parents=True)
raw_data_folder.parent.mkdir(exist_ok=True, parents=True)
if not raw_data_folder.exists():
    raw_data_folder.symlink_to(nas2_raw_data_folder)

# %% [markdown]
# ### Download genotypes

# %% tags=[]
main_url = "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE196nnn/GSE196829/suppl/"

# %% tags=[]
file = "GSE196829_onek1k_genotype_matrix.txt.gz"

# %% tags=[]
# !wget -O {raw_data_folder}/"genotype_matrix.txt.gz" {main_url}/{file}

# %% tags=[]
file = "GSE196829_RAW.tar"

# %% tags=[]
# !wget -O {raw_data_folder}/{file} {main_url}/{file}

# %% [markdown]
# ### Download and convert idat files

# %% tags=[]
idat_folder = raw_data_folder/"idat"

# %% tags=[]
# make idat files
os.environ["idat_folder"] = str(idat_folder)
# # !mkdir -p $idat_folder
# # !tar -xf {raw_data_folder}/{file} -C $idat_folder

# %% tags=[]
# unzip
# %%bash
pushd ${idat_folder}
for f in *.gz ; do gunzip -c "$f" > "${f%.*}" ; done

# %% tags=[]
file.parent / "_".join(split[1:])

# %% tags=[]
# convert to correct idat file name format as per https://github.com/freeseek/gtc2vcf
# we remove the 1st sample id, which was probably added by GEO
# we will map everything later
for file in idat_folder.iterdir():
    if file.name.endswith("idat") and file.name.count("_") == 3:
        split = file.name.split("_")
        file.rename(file.parent / "_".join(split[1:]))

# %% [markdown]
# ### Download genotype sample metadata

# %% tags=[]
# !wget -P {raw_data_folder} https://ftp.ncbi.nlm.nih.gov/geo/series/GSE196nnn/GSE196829/matrix/GSE196829_series_matrix.txt.gz

# %% [markdown]
# ### Download Illumina reference

# %% [markdown]
# Download reference
# https://support.illumina.com/array/array_kits/infinium-global-screening-array/downloads.html

# %% tags=[]
# !wget -P {raw_data_folder} https://webdata.illumina.com/downloads/productfiles/global-screening-array/v2-0/infinium-global-screening-array-24-v2-0-a2-manifest-file-csv.zip

# %% tags=[]
# !wget -P {raw_data_folder} https://webdata.illumina.com/downloads/productfiles/global-screening-array/v2-0/infinium-global-screening-array-24-v2-0-a2-manifest-file-bpm.zip

# %% tags=[]
# !wget -P {raw_data_folder} https://webdata.illumina.com/downloads/productfiles/global-screening-array/v2-0/gsa-24-v2-0-A1-cluster-file.zip

# %% tags=[]
# !unzip -d {raw_data_folder} {raw_data_folder}/infinium-global-screening-array-24-v2-0-a2-manifest-file-bpm.zip
# !unzip -d {raw_data_folder} {raw_data_folder}/gsa-24-v2-0-A1-cluster-file.zip

# %% tags=[]
# !unzip -d {raw_data_folder} {raw_data_folder}/infinium-global-screening-array-24-v2-0-a2-manifest-file-csv.zip

# %% [markdown]
# ### Run IAAP

# %% [markdown]
# https://github.com/freeseek/gtc2vcf

# %% tags=[]
import os

# %% tags=[]
bpm_manifest_file = raw_data_folder/"GSA-24v2-0_A2.bpm"
egt_cluster_file = raw_data_folder/"GSA-24v2-0_A1_ClusterFile.egt"

idat_folder = str(raw_data_folder/"idat")

gtc_folder = raw_data_folder/"gtc"
gtc_folder.mkdir(exist_ok = True, parents = True)

# %% tags=[]
os.environ["bpm_manifest_file"] = str(bpm_manifest_file)
os.environ["egt_cluster_file"] = str(egt_cluster_file)
os.environ["gtc_folder"] = str(gtc_folder)

# %% tags=[] language="bash"
# $HOME/bin/iaap-cli/iaap-cli gencall --help

# %% tags=[] language="bash"
# CLR_ICU_VERSION_OVERRIDE="$(uconv -V | sed 's/.* //g')" LANG="en_US.UTF-8" $HOME/bin/iaap-cli/iaap-cli \
#   gencall ${bpm_manifest_file} $egt_cluster_file ${gtc_folder} \
#   --idat-folder $idat_folder \
#   --output-gtc \
#   --gender-estimate-call-rate-threshold -0.1 \
#   --num-threads 10

# %% [markdown]
# ### Illumina GTC to VCF

# %% tags=[]
csv_manifest_file = raw_data_folder/"GSA-24v2-0_A2.csv"
egt_cluster_file = raw_data_folder/"GSA-24v2-0_A1_ClusterFile.egt"

gtc_folder = raw_data_folder/"gtc"

ref = pathlib.Path("/home/wsaelens/GRCh38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna")

vcf_folder = raw_data_folder/"vcf"
vcf_folder.mkdir(exist_ok = True, parents = True)

# %% tags=[]
os.environ["csv_manifest_file"] = str(csv_manifest_file)
os.environ["egt_cluster_file"] = str(egt_cluster_file)
os.environ["gtc_folder"] = str(gtc_folder)
os.environ["vcf_folder"] = str(vcf_folder)
os.environ["ref"] = str(ref)

# %% tags=[]
os.environ["BCFTOOLS_PLUGINS"] = "/home/wsaelens/bin"

# %% tags=[] language="bash"
# echo $bpm_manifest_file
# echo $csv_manifest_file
# echo $egt_cluster_file
# echo $gtc_folder
# echo $ref
# echo $vcf_folder
# $HOME/bin/bcftools +gtc2vcf \
#   --no-version -Ou \
#   --bpm $bpm_manifest_file \
#   --csv $csv_manifest_file \
#   --egt $egt_cluster_file \
#   --gtcs $gtc_folder \
#   --fasta-ref $ref \
#   --extra $vcf_folder/out.tsv | \
#   $HOME/bin/bcftools sort -Ou -T ./bcftools. | \
#   $HOME/bin/bcftools norm --no-version -Ob -c x -f $ref | \
#   tee $vcf_folder/out.bcf | \
#   $HOME/bin/bcftools index --force --output $vcf_folder/out.bcf.csi

# %% tags=[]
# !ls {vcf_folder}

# %% [markdown]
# ### Further post-processing

# %% tags=[]
data_folder = raw_data_folder.parent
data_folder.mkdir(exist_ok = True)

# %% tags=[]
# !cp {raw_data_folder}/vcf/out.bcf {data_folder}/out.bcf

# %% tags=[]
# !cp {raw_data_folder}/vcf/out.bcf.csi {data_folder}/out.bcf.csi

# %% [markdown]
# ### Filter samples and clean

# %% tags=[]
idat_folder = raw_data_folder/"idat"
samples = []
for file in idat_folder.iterdir():
    if file.name.endswith("idat.gz") and file.name.count("_") == 3:
        old_sample_name = file.name.split("_")[1] + "_" + file.name.split("_")[2]
        samples.append({"old":old_sample_name, "gsm":file.name.split("_")[0], "gsm_ix":int(file.name.split("_")[0][3:])})
samples = pd.DataFrame(samples)
samples = samples.groupby("gsm_ix").first().reset_index().sort_values("gsm_ix")

# %% tags=[]
genotype_metadata = pd.read_table(raw_data_folder/"GSE196829_series_matrix.txt.gz" , skiprows=29, index_col=0).T
genotype_metadata["gsm"] = genotype_metadata["!Sample_geo_accession"]
genotype_metadata["donor_id"] = genotype_metadata["!Sample_characteristics_ch1"].iloc[:, 0].str.split(": ").str[1]
genotype_metadata = genotype_metadata.loc[genotype_metadata["donor_id"] != "Blood"].reset_index(drop = True)

# %% tags=[]
samples["donor"] = genotype_metadata[["gsm", "donor_id"]].set_index("gsm")["donor_id"].reindex(samples["gsm"]).values

# %% tags=[]
samples = samples.loc[~pd.isnull(samples["donor"])]

# %% tags=[]
samples[["old"]].to_csv("samples.csv", index = False, header = False)

# %% tags=[]
cleaned_bcf_file = data_folder/"out_cleaned.bcf"

# %% tags=[]
# !bcftools view --threads 10 -S samples.csv {data_folder}/out.bcf | bcftools annotate --threads=10 -x FILTER,FORMAT | bcftools view --threads=10 -o {cleaned_bcf_file}

# %% tags=[]
# samples[["old", "donor"]].to_csv("samples.csv", index = False, header = False)

# %% tags=[]
# # !bcftools reheader --threads 10 -s samples.csv -o {data_folder}/out_filtered3.bcf {data_folder}/out_filtered2.bcf

# %% tags=[]
# !bcftools index --threads 10 {cleaned_bcf_file} --force

# %% [markdown]
# ### Imputation

# %% tags=[]
cleaned_bcf_file = data_folder/"out_cleaned.bcf"

# %% [markdown]
# Create imputation files

# %% tags=[]
for_imputation_folder = data_folder/"for_imputation"
for_imputation_folder.mkdir(exist_ok = True)

# %% tags=[]
os.environ["cleaned_bcf_file"] = str(data_folder/"out_cleaned.bcf")
os.environ["for_imputation_folder"] = str(final_bcf_folder)

# %% tags=[] language="bash"
# bcftools index -s $cleaned_bcf_file | cut -f 1 | while read C; do bcftools view --threads=10 -O z -q 0.01:minor -o $for_imputation_folder/${C}.vcf.gz $cleaned_bcf_file "${C}" ; done

# %% [markdown]
# ![image.png](attachment:1b06fe6f-b0db-44d3-8de6-c489b86f6d2e.png)

# %% tags=[]
after_imputation_folder = data_folder/"after_imputation"
after_imputation_folder.mkdir(exist_ok = True)

# %% tags=[]
os.environ["after_imputation_folder"] = str(after_imputation_folder)

# %% language="bash"
# pushd $after_imputation_folder
# curl -sL https://imputationserver.sph.umich.edu/get/3060183/19e4dc2d3aa55b79281ac831c2a69bfb1b6539b201e3fb50dbeacd5450c228f4 | bash

# %% tags=[] language="bash"
# pushd $after_imputation_folder
# curl -sL https://imputationserver.sph.umich.edu/get/3060189/15181615a9143ec612dac501bdf51ee45ee3c1b180d2e58b82127158443749ca | bash

# %% [markdown]
# ## Download expression

# %% tags=[]
main_url = "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE196nnn/GSE196735/suppl/"

# %% tags=[]
file = "GSE196735_RAW.tar"

# %% tags=[]
# !wget -O {raw_data_folder}/"GSE196735_RAW.tar" {main_url}/{file}

# %% tags=[]
# !curl -o {raw_data_folder}/"cellxgene_normal.tar" "https://corpora-data-prod.s3.amazonaws.com/594bcd4b-e3fa-48b4-871f-f99f77429888/local.h5ad?AWSAccessKeyId=ASIATLYQ5N5X6O3IWU27&Signature=KF8DhN4tE3RuJVG%2Fig3E6EzIccs%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEJX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLXdlc3QtMiJIMEYCIQCYmGZ9rBYeyQXd7HFW3djmsHNYisrpsn5a9xPPb8Lm1QIhANemN%2FdwcymTD%2F%2F%2FtFvxdNTlAbNb3NeBtS9HyyI2v8WOKusDCD4QARoMMjMxNDI2ODQ2NTc1IgyJ1rr19%2FpP8b9CFpAqyANfbPtS29FCfXwmlXuBWxecJpsU3h67FIQDfPq94Cj0Ja3R6YfQfBuY6h31MZu4Ffwfs8Jt%2B2Nr0SLhPQGBjKJgwfC%2Fu6emkyPB9nSTiccTpfUNJp%2F%2F2x72N2rrEB0I%2FredJRbGaBj3EuEkB%2Bt9upwkfzpmojsz9jZE1QquHwDr0PGDNZ1OYCVtO%2B55aSW1Vfba6RWMgx9qctCJkOWgRv0xBq2ABEO1DZQj9ZVOFEHMp%2FzV7%2BBLX8JGdNHMZVJtYa2W0GRNkt%2FYgo7RcCNtorKzA%2FVpY1Xl%2F14wn5bxpdfEG2jWESZHyvVjonH5EN3IUjUeam85fR7zKirdZgvZXR7IPgiXQ%2B7HQtK9GXi1YgUzgvgLvRhU92A1dZknLKGvVWLVFdCB5SO%2Ba5ah%2BfnpuIH1QAyNYkqkeCgE%2F%2BiU0CC6BP5ouZU3ICRHCsJeylpqayFLpbWUTDh%2Fyc08v86wVDi4L7jmMkvBnp8eMzmkMfhhDdXgWmPHG7itDZa2GWgxLDB9EH4MgePboqCF3IXN3dZHMzgqChCeYuMbXDIIlDIZ%2B5fTT%2BHLOWl1kpWQ8j789sipiX7etjFkAgGyMMJ3qqo4d7y2jUNk9lcwhcPrnwY6pAEJuwD8DfFCFy88T0r%2F3r4LmW8flo4bCpiZU%2BEKZ6s6gcV4m%2BbsuIRcn9dJ1xbNNE2rE30ULRglrD7GTeuQ3yaO1pac1EEYrNlQjHACGiZEL%2B4K%2FQxaP6zNHMl4FxJV7LknNLkTzGcJmuzw4ydb5%2FXJ9k02KzDo3A8soSEGE%2FYD0i9TiX34bEYsaGBNVrEuK39T5dqA5cUd4jOXY1g5UPszHUhUWQ%3D%3D&Expires=1677994368"

# %% tags=[]
# !curl -o {raw_data_folder}/"cellxgene_lupus.tar" "https://corpora-data-prod.s3.amazonaws.com/0354fbcd-397c-4ec6-9013-97aa36c712b7/local.h5ad?AWSAccessKeyId=ASIATLYQ5N5XUNT2KGHG&Signature=GQ33wg62Z%2BEbHPS9LIBeqyP3btU%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEJP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLXdlc3QtMiJIMEYCIQDRM8nWl9MgiQZl23IFRY83GgpuNB3U6YZ30HHE71PhZwIhALc73thiRR4N%2F9RvrMTFO9HFJ8askXLVg1VwW0YeqBHGKusDCDwQARoMMjMxNDI2ODQ2NTc1IgzDzUj1PW7EEImiV3QqyAMdMcIbz5fqXx4mQo1hbfzjJLLPVg3JfzIrw5MqRE8B4axjh2npNY7JnTqxz2TZ3p2sdvpgiA1ob7rwvZdCT%2FsntGmnVO8d17oTDr8J2ND7inWFb3ILEaoAG9%2FBEkkWMgl4Phic5Kjobefrh6KDkeGmW%2FIUYdK7uYhkTD0OkpoRF1bLRD6XaDOU0Ty2bvXaD14sRgf%2FHROO0Q0foDpDJ%2BdBGN90cNy8E9qFNldRTt7HjXTQJZChqlhVKGBrCYEcM8zaGoXHDbsSfwoDWATP9t0wbfwM9PZqxcYefeno9HobGp1tWXW4icUi8wwa%2FXaZOpE7gwpJ63MVy9TPlhYVmADkL%2F4XIcXsRE9bwnZdtJvb%2BMtNTYEeMwxlbDOBdfWb1Ho3UCVhYdTmK6Fgm512QGqF4B0sR9td4l2B%2FXkDt1N4t6ZydZz%2BKeLvXzAUYT3fzzAGqOqT6a%2BO%2Bq%2BOB%2B1Jih6oPyvZPCZCNxZWWvFJ3OcDTiVG%2FtWex7rG7mppqHH8H250V7XPj3oUS8Q5InglUfLKvQJDgvXaObEGhocCFkYONOLFKToYpaTZLvmEMrZLUopW3VRER2%2BHIBxdT7BHq4ZL2ngBRu5zUR0wgZfrnwY6pAGpoAmIlEhAYChjy8zEtWxfzgbEWgcVs98%2By%2B0txmZl2zgCJUK4Jw%2FnDp9lZ6Ri%2Bs5cSYT9OQWohg6JnrKUWQJzRMPI4m2DyJTjblM9w2J12g12UipE06F7hguDO%2Fz6Uq0MOCFFgAoKvvPXlwof42NsXwBDNCNBbcrUXHY6%2BspCYNjfRMfd%2BTC5Y%2B%2BoswuJlMA83oLb%2FNMdJdsJIntS0jnPkxiMlQ%3D%3D&Expires=1677994400"
