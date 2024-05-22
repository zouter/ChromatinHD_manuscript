# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: chromatinhd
#     language: python
#     name: python3
# ---

# %%
from IPython import get_ipython

if get_ipython():
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")

import pickle

import tqdm.auto as tqdm

import pathlib

import polars as pl

# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

# %%
# download latest release (2.0) here http://www.mulinlab.org/causaldb/index.html
# put in data/qtl/hs/causaldb

# %%
folder_qtl = chd.get_output() / "data" / "qtl" / "hs" / "causaldb"
folder_qtl.mkdir(exist_ok=True, parents=True)

# %%
# !ls /home/wsaelens/projects/chromatinhd/chromatinhd_manuscript/output/data/qtl/hs/causaldb

# %%
if not (folder_qtl / "v2.0").exists():
    # !tar -xzf {folder_qtl}/credible_set.v2.0.20230628.tar.gz -C {folder_qtl}

# %%
# !ls -lh {folder_qtl}

# %%
# !cat {folder_qtl}/v2.0/README.md

# %%
qtl = pd.read_table(folder_qtl / "v2.0" / "credible_set.txt")
qtl = qtl.loc[qtl["rsid"] == qtl["lead_snp"]].copy()

# %%
meta = pd.read_table(folder_qtl / "v2.0" / "meta.txt").set_index("meta_id")

# %%
qtl["disease/trait"] = pd.Categorical(qtl["meta_id"].map(meta["trait"]))

# %%
qtl.query("lead_snp == 'rs6592965'").iloc[0]

# %%
trait_counts = qtl["disease/trait"].value_counts()
trait_counts.loc[trait_counts.index.str.contains("iron")]

# %%
traits_oi = pd.DataFrame(
    [
        ["Multiple sclerosis"],
        ["Type 1 diabetes"],
        ["Inflammatory bowel disease"],
        ["Crohn's disease"],
        ["Systemic lupus erythematosus"],
        ["Rheumatoid arthritis"],
        ["Ankylosing spondylitis"],
        ["Hodgkin's lymphoma"],
        ["Psoriasis"],
        ["Post bronchodilator FEV1/FVC ratio in COPD"],
        ["Non-Hodgkin's lymphoma"],
        ["Core binding factor acute myeloid leukemia"],
        ["Chronic lymphocytic leukemia"],
        ["Interleukin-6 levels"],
        ["Interleukin-18 levels"],
        [
            "6-month creatinine clearance change response to tenofovir treatment in HIV infection (treatment arm interaction)"
        ],
        [
            "Time-dependent creatinine clearance change response to tenofovir treatment in HIV infection (time and treatment arm interaction)"
        ],
        ["Autoimmune thyroid disease"],
        ["IgG glycosylation"],
        ["Asthma"],
        ["Allergic disease (asthma, hay fever or eczema)"],
        ["High IL-1beta levels in gingival crevicular fluid"],
        ["C-reactive protein levels (MTAG)"],
        ["Behcet's disease"],
        ["Neutrophil count"],
        ["Eosinophil counts"],
        ["Monocyte count"],
        ["Lymphocyte count"],
        ["Endometriosis"],
        ["Medication use (thyroid preparations)"],
        ["Basophil count"],
        [
            "Acute graft versus host disease in bone marrow transplantation (recipient effect)"
        ],
        ["Selective IgA deficiency"],
        ["Lymphocyte-to-monocyte ratio"],
        ["COVID-19"],
        ["C-reactive protein"],
    ],
    columns=["disease/trait"],
).set_index("disease/trait")
motifscan_name = "causaldb_immune"


traits_oi = pd.DataFrame([
    ["Chronic lymphocytic leukemia"],
    ["Acute lymphoblastic leukemia (childhood)"],
    ["Hodgkin's lymphoma"],
    ["Childhood ALL/LBL (acute lymphoblastic leukemia/lymphoblastic lymphoma) treatment-related venous thromboembolism"],
    ["B-cell malignancies (chronic lymphocytic leukemia, Hodgkin lymphoma or multiple myeloma) (pleiotropy)"],
    ["Non-Hodgkin's lymphoma"],
    ["Hodgkin lymphoma"],
    ["Non-Hodgkins lymphoma"],
    ["C85 Other and unspecified types of non-Hodgkin's lymphoma"],
    ['Non-follicular lymphoma'],
    ['Large cell lymphoma'],
    ['Non-Hodgkins lymphoma'],
    ['Diffuse large B-cell lymphoma (controls excluding all cancers)'],
    ["C83 Diffuse non-Hodgkin's lymphoma"],
    ['Hodgkin lymphoma (controls excluding all cancers)'],
    ['Cancer code, self-reported: non-hodgkins lymphoma'],
    ['Other and unspecified types of non-Hodgkin lymphoma'],
    ['Waldenstrom macroglobulinemia, lymphoplasmacytic lymphoma (controls excluding all cancers)'],
    ['Cancer code, self-reported: hodgkins lymphoma / hodgkins disease'],
    ['Non-follicular lymphoma (controls excluding all cancers)'],
    ['Other and unspecified types of non-Hodgkin lymphoma (controls excluding all cancers)'],
    ['Hodgkin lymphoma']
], columns = ["disease/trait"]).set_index("disease/trait")
motifscan_name = "causaldb_lymphoma"

traits_oi = pd.read_table(chd.get_output() / "data" / "qtl/hs/gwas" / f"traits_gwas_liver.tsv", index_col = 0)
traits_oi = pd.concat([
    traits_oi,
    trait_counts.loc[trait_counts.index.str.contains("liver")].sort_values(ascending = False).head(10).to_frame(name = "n"),
    trait_counts.loc[trait_counts.index.str.contains("gall")].sort_values(ascending = False).head(10).to_frame(name = "n"),
    trait_counts.loc[trait_counts.index.str.contains("Liver")].sort_values(ascending = False).head(10).to_frame(name = "n"),
    trait_counts.loc[trait_counts.index.str.contains("iron")].sort_values(ascending = False).head(10).to_frame(name = "n"),
])
motifscan_name = "causaldb_liver"

# %%
qtl_mapped = qtl.loc[qtl["disease/trait"].isin(traits_oi.index)].copy()
qtl_mapped["snp"] = qtl_mapped["rsid"]
qtl_mapped = qtl_mapped[["snp", "disease/trait"]]

# %%
qtl_mapped = qtl_mapped.drop_duplicates(
    ["snp", "disease/trait"]
)

# %%
qtl_mapped.to_pickle(folder_qtl / ("qtl_mapped_" + motifscan_name + ".pkl"))

# %% [markdown]
# ### Get SNP Info

# %%
snp_info_file = pathlib.Path(chd.get_output() / "snp_info.pkl")
if not snp_info_file.exists():
    snp_info = pd.DataFrame(
        {
            "snp": pd.Series(dtype=str),
            "chr": pd.Series(dtype=str),
            "start": pd.Series(dtype=int),
            "end": pd.Series(dtype=int),
        }
    ).set_index("snp")
    pickle.dump(snp_info, snp_info_file.open("wb"))
snp_info = pickle.load(snp_info_file.open("rb"))
len(snp_info)

# %%
snps_missing = pd.Index(qtl_mapped["snp"].unique()).difference(snp_info.index)

# %%
len(snps_missing)

# %%
import sqlalchemy as sql
import pymysql

# %%
n = 1000

# %%
chunks = [snps_missing[i : i + n] for i in range(0, len(snps_missing), n)]

# %%
len(chunks)

# %%
for snps in tqdm.tqdm(chunks):
    snp_names = ",".join("'" + snps + "'")
    query = f"select * from snp151 where name in ({snp_names})"
    result = pd.read_sql(
        query,
        "mysql+pymysql://genome@genome-mysql.cse.ucsc.edu/{organism}?charset=utf8mb4".format(
            organism="hg38"
        ),
    ).set_index("name")
    result = result.rename(
        columns={
            "chrom": "chr",
            "chromStart": "start",
            "chromENd": "end",
            "name": "snp",
        }
    )

    result["start"] = result["start"].astype(int)
    result["end"] = result["start"].astype(int)
    result.index.name = "snp"
    result = result.groupby("snp").first()
    result = result.reindex(snps)
    result.index.name = "snp"

    # assert result.index.str.contains(";").any()

    snp_info = pd.concat([snp_info, result], axis=0)
    pickle.dump(snp_info, snp_info_file.open("wb"))

# %% [markdown]
# ##

# %%
snp_info = pickle.load((chd.get_output() / "snp_info.pkl").open("rb"))
# qtl_mapped = pd.read_pickle(folder_qtl / ("qtl_mapped_" + motifscan_name + ".pkl"))
# qtl_mapped.index = np.arange(len(qtl_mapped))
association = qtl_mapped.join(snp_info, on="snp")
association = association.loc[~pd.isnull(association["start"])]
association["pos"] = association["start"].astype(int)

# %%

# %%
