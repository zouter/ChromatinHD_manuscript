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

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")

import pickle

import scanpy as sc
import eyck

import tqdm.auto as tqdm
import io
import pathlib

# %%
fs4_folder = pathlib.Path("/home/wouters/fs4")

# stead-state - 8um
# folder = fs4_folder / "exp/units/u_mgu/transfer/VisiumHD/SCA002_reseq/outs/binned_outputs/square_008um/"

# steady-state - 2um
folder = fs4_folder / "exp/units/u_mgu/transfer/VisiumHD/SCA002_reseq/outs/binned_outputs/square_002um/"

# metastasis - 8um
# folder = fs4_folder / "exp/units/u_mgu/private/singlecell/SCA006_reseq/outs/binned_outputs/square_008um/"

# metastasis - 2um
# folder = fs4_folder / "exp/units/u_mgu/private/singlecell/SCA006_reseq/outs/binned_outputs/square_002um/"

# %%
adata = sc.read_10x_h5(
    folder / "filtered_feature_bc_matrix.h5"
)
adata.var["symbol"] = adata.var.index
adata.var["gene"] = adata.var["gene_ids"]
adata.var.index = adata.var["gene"]
adata.var = adata.var.drop(columns=["gene_ids", "gene"])

# %%
print(f"""
Data: {adata.X.data.shape[0] * 32 / 8 / 1024 / 1024 / 1024:.2f}Gb
Indptr: {adata.X.indptr.shape[0] * 8 / 1024 / 1024:.2f}Mb
Indices: {adata.X.indices.shape[0] * 8 / 1024 / 1024/1024:.2f}Gb
""")

# %%
tissue_positions = pd.read_parquet(
    folder / "spatial/tissue_positions.parquet"
).set_index("barcode")

# %%
# import skimage

# X2 = skimage.measure.block_reduce(adata.X.todense(), (10, 10), np.sum)

# %%
adata.obsm["spatial"] = tissue_positions.loc[
    adata.obs.index, ["array_col", "array_row"]
].values


# %%
adata.obs["n_counts"] = adata.X.sum(axis=1).A1
adata.obs["log10n_counts"] = np.log10(adata.X.sum(axis=1).A1 + 1)

# %%
eyck.modalities.transcriptome.plot_embedding(
    adata,
    [
        # "log10n_counts",
        # "Hal",
        "Fcgr1",
        "Mertk",
        "Adgre1",
        # "Glul",
        # "Xcr1",
        # "Cd209a",
        # "Ccl21a",
        # "Cd3e",
        # "Ccl21a",
        # "Mki67",
        # "Pcna",
        # "Clec4f",
        # "Cyp2f2",
        # "Lrat",
        # "Glul",
        # "Vwf",
        # "Col1a1",
        # "Adgre1",
        # "Lhx2",
        # "Gdf2",
        # "Dll4",
        # "Ngf",
        # "Ncam1",
        # "Irf4",
        # "Ocstamp",
        # "Dcstamp",
        # "Cyp2e1",
        # "Epcam",
        # "Vwf",
        # "Flt3",
        # "Ccr2",
        # "Mecom",
        # "Ccl21a",
        # "Lyve1",
        # "Pcdh17",
        # "Stab2",
    ],
    embedding="spatial",
    panel_size=10,
).display()

# %%
data = pd.DataFrame(
    {
        g: sc.get.obs_df(adata, eyck.modalities.transcriptome.gene_id(adata.var, g))
        for g in [
            "Clec4f",
            "Ngfr",
            "Lrat",
            "Lama1",
            "Ptger2",
            "Glul",
            "Cyp2f2",
            "Cyp2e1",
            "Cd5l",
            "Lyve1",
            "Pcdh17",
            "Stab2",
        ]
    }
)

# %%
data["hsc"] = data[["Ngfr", "Lrat", "Lama1", "Ptger2"]].sum(axis=1)
data["endo"] = data[["Lyve1", "Pcdh17", "Stab2"]].sum(axis=1)

# %%
fig, ax = plt.subplots(figsize=(2, 2))
ax.scatter(data["Clec4f"], data["hsc"], s=1, alpha=0.1)
# ax.scatter(data["Clec4f"], data["Ngfr"], s=1, alpha=0.1)

# %%


# %%
cors = np.corrcoef(data.T)
np.fill_diagonal(cors, 0)
sns.heatmap(pd.DataFrame(cors, index=data.columns, columns=data.columns), annot=True)

# %%
folder = fs4_folder / "exp/units/u_mgu/transfer/VisiumHD/SCA002_reseq/outs/"
h5_file_location = folder / "feature_slice.h5"

# %%
import h5py as h5
import numpy as np
import json
import dataclasses

ROW_DATASET_NAME = "row"
COL_DATASET_NAME = "col"
DATA_DATASET_NAME = "data"

METADATA_JSON_ATTR_NAME = "metadata_json"
UMIS_GROUP_NAME = "umis"
TOTAL_UMIS_GROUP_NAME = "total"

@dataclasses.dataclass
class CooMatrix:
    row: list[int]
    col: list[int]
    data: list[int | float]

    @classmethod
    def from_hdf5(cls, group):
        return cls(
            row=group[ROW_DATASET_NAME][:],
            col=group[COL_DATASET_NAME][:],
            data=group[DATA_DATASET_NAME][:],
        )

    def to_ndarray(self, nrows, ncols, binning_scale = 1):
        """Convert the COO matrix representation to a dense ndarray at the specified binning scale."""
        ncols_binned = int(np.ceil(ncols / binning_scale))
        nrows_binned = int(np.ceil(nrows / binning_scale))

        result = np.zeros((nrows_binned, ncols_binned), dtype="int32")
        for row, col, data in zip(self.row, self.col, self.data):
            result[row // binning_scale, col // binning_scale] += data
        return result

# Load total UMIs at 8um bin size
with h5.File(h5_file_location, "r") as h5_file:
    metadata = json.loads(h5_file.attrs[METADATA_JSON_ATTR_NAME])
    umis_8um = CooMatrix.from_hdf5(h5_file[UMIS_GROUP_NAME][TOTAL_UMIS_GROUP_NAME]).to_ndarray(
        nrows=metadata["nrows"], ncols=metadata["ncols"], binning_scale=4
    )
# %%
umis_8um