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
import chromatinhd as chd
import chromatinhd_manuscript as chdm
import gimmedata

import pandas as pd

# %%
dataset_preproc_folder = chd.get_output() / "data" / "hmmc"
dataset_preproc_folder.mkdir(parents=True, exist_ok=True)

# %%
# Your task is to download the multiome dataset from GSE219015, utilizing `dataset_preproc_folder` as your preprocessing folder.

# %%
# I will look at the paper to see what type of data is likely submitted

# %%
# I process the GSE series data
series = gimmedata.process_series("GSE219015")

# %%
# Are there any subseries?
series.subseries

# %%
# There are several subseries, it is often easier to download data there. I will look at the first subseries
subseries = gimmedata.process_series(series.subseries["series"][0])

# %%
# I check out the supplementary file names
subseries.supplementary_files["filename"].tolist()

# %%
# I see that this subseries contains the raw data files for both the BMMC and HSPC cells in separate files, which probably means these were coming from separate libraries. `atac_fragments.tsv.gz`` is the ideal file for the ATAC-seq data, although I will have to ensure to check the genome to which this was mapped. `filtered_feature_bc_matrix.h5`` is the ideal format for the scRNA-seq data, although I will have to check which gene symbols were used. There are also mtDNA Variants matrix files. I do not yet know how to process these.

# %%
# I check every subseries and join all the supplementary files, all the while creating a dataframe containing the name of each file which will be used to reconstruct the samples
files = []

for subseries in series.subseries["series"]:
    print(subseries)
    subseries = gimmedata.process_series(subseries)
    for _, supplementary_file in subseries.supplementary_files.iterrows():
        file_path = dataset_preproc_folder / supplementary_file["filename"]
        if not file_path.exists():
            print(file_path)
            urllib.request.urlretrieve(supplementary_file["url"], file_path)

        files.append({"subseries": subseries.title, "filename": supplementary_file["filename"]})
files = pd.DataFrame(files)

# %%
# I now want to extract the samples. If possible, I want to do that using the filenames. I observe that each filename starts with a GSE number, followed by some info on the samples, followed by a dot which indicates what is contained within the file, ending with the file extension. As such, I can extract the sample name by simply splitting the filename by a dot, and the sample is just the first split. 
samples = []
for filename in files["filename"]:
    split_filename = filename.split(".")
    samples.append({"sample": split_filename[0]})
samples = pd.DataFrame(samples).drop_duplicates(keep = "first").set_index("sample")
print(samples.index.tolist())

# %%
# I see that some samples do not have the sample info. I remove these for now.
samples = samples[~samples.index.str.contains("RAW")]

# %%
# I know extract the relevant information: the patient and the sorting, from the filename.
# I note that sometimes, I observe a T2. Using information from the paper and the fact that it starts with a t, I infer that this likely indicates the second time point. I will store that information in a separate column.
samples["sorting"] = samples.index.str.split("_").str[-1]
samples["patient"] = samples.index.str.split("_").str[1]
samples["timepoint"] = (samples.index.str.contains("T2")).astype(int)

# %%
samples
