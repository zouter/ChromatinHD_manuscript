# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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

import tqdm.auto as tqdm

import pathlib

# %%
import chromatinhd as chd

# import chromatinhd_manuscript as chdm
# from manuscript import Manuscript

# manuscript = Manuscript(chd.get_git_root() / "manuscript")

# %%
folder_qtl = chd.get_output() / "data" / "qtl" / "hs" / "gwas"
folder_qtl.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# ## Download

# %%
import requests

# %%
# !curl --location https://www.ebi.ac.uk/gwas/api/search/downloads/full > {folder_qtl}/full.tsv

# %%
# !head -n 5 {folder_qtl}/full.tsv

# %%
# !wc -l {folder_qtl}/full.tsv

# %% [markdown]
# ## OBO

# %%
# # !pip install owlready2
# if not (folder_qtl / "efo.owl").exists():
    # !wget http://www.ebi.ac.uk/efo/efo.owl -O {folder_qtl}/efo.owl

# %%
import owlready2

# %%
onto = owlready2.get_ontology(str(folder_qtl / "efo.owl")).load()

# %%
parent = onto.search(iri="http://www.ebi.ac.uk/efo/EFO_0000540")[0] # immune system disease
parent = onto.search(iri="http://www.ebi.ac.uk/efo/EFO_0000589")[0] # metabolic disease
parent = onto.search(iri="http://www.ebi.ac.uk/efo/EFO_0001421")[0] # liver disease

# %%
children = onto.search(subclass_of=parent)

# %%
rdf = onto.get_namespace("http://www.w3.org/2000/01/rdf-schema#")
obo = onto.get_namespace("http://purl.obolibrary.org/obo/")

# %%
traits = []
for child in children:
    description = obo.IAO_0000115[child][0] if len(obo.IAO_0000115[child]) else ""
    traits.append({"trait": rdf.label[child][0], "description": description})
traits = pd.DataFrame(traits)

# %% [markdown]
# ## Process

# %%
qtl = pd.read_table(folder_qtl / "full.tsv", index_col=None)
qtl = qtl.rename(columns=dict(zip(qtl.columns, [col.lower() for col in qtl.columns])))

# %%
qtl["strongest_risk_allele"] = qtl["strongest snp-risk allele"].str.split("-").str[-1]

# %%
diseases = qtl.groupby("disease/trait").size().sort_values(ascending=False)

# %%
trait_counts = qtl["disease/trait"].value_counts()

#%%
trait_counts.loc[trait_counts.index.str.contains("asthma", flags = re.IGNORECASE)].head(20)

# %%
import re
# traits_oi = pd.DataFrame([
#     *trait_counts.loc[trait_counts.index.str.contains("brain", flags = re.IGNORECASE)][:20].index,
#     *trait_counts.loc[trait_counts.index.str.contains("neuroblastoma", flags = re.IGNORECASE)][:10].index,
#     *trait_counts.loc[trait_counts.index.str.contains("alzheimer", flags = re.IGNORECASE)][:10].index,
#     *trait_counts.loc[trait_counts.index.str.contains("microglia", flags = re.IGNORECASE)][:10].index,
#     *trait_counts.loc[trait_counts.index.str.contains("parkinson", flags = re.IGNORECASE)][:10].index,
#     *trait_counts.loc[trait_counts.index.str.contains("schizophrenia", flags = re.IGNORECASE)][:10].index,
#     *trait_counts.loc[trait_counts.index.str.contains("bipolar", flags = re.IGNORECASE)][:10].index,
#     *trait_counts.loc[trait_counts.index.str.contains("autism", flags = re.IGNORECASE)][:10].index,
#     *trait_counts.loc[trait_counts.index.str.contains("attention-deficit")][:10].index,
#     *trait_counts.loc[trait_counts.index.str.contains("sclerosis", flags = re.IGNORECASE)][:10].index,
#     *trait_counts.loc[trait_counts.index.str.contains("epilepsy", flags = re.IGNORECASE)][:10].index,
#     *trait_counts.loc[trait_counts.index.str.contains("migraine", flags = re.IGNORECASE)][:10].index,
# ], columns = ["disease/trait"]).set_index("disease/trait"); motifscan_name = "gwas_cns"

# traits_oi = pd.DataFrame([[x] for x in [
#     "Cholangiocarcinoma in primary sclerosing cholangitis",
#     "Cholangiocarcinoma in primary sclerosing cholangitis (time to event)",
#     *trait_counts.loc[trait_counts.index.str.contains("Liver")][:10].index,
#     *trait_counts.loc[trait_counts.index.str.contains("liver")][:10].index,
#     *trait_counts.loc[trait_counts.index.str.contains("Cholesterol")][:10].index,
#     *trait_counts.loc[trait_counts.index.str.contains("glucose")][:5].index,
#     *trait_counts.loc[trait_counts.index.str.contains("Hepat")][:5].index,
#     *trait_counts.loc[trait_counts.index.str.contains("hepat")][:5].index,
#     "Lipid metabolism phenotypes",
# ]], columns = ["disease/trait"]).set_index("disease/trait")
# motifscan_name = "gwas_liver"


# traits_oi = pd.DataFrame([
#     ["Chronic lymphocytic leukemia"],
#     ["Acute lymphoblastic leukemia (childhood)"],
#     ["Hodgkin's lymphoma"],
#     ["Childhood ALL/LBL (acute lymphoblastic leukemia/lymphoblastic lymphoma) treatment-related venous thromboembolism"],
#     ["B-cell malignancies (chronic lymphocytic leukemia, Hodgkin lymphoma or multiple myeloma) (pleiotropy)"],
#     ["Non-Hodgkin's lymphoma"],
# ], columns = ["disease/trait"]).set_index("disease/trait")
# motifscan_name = "gwas_lymphoma"

# traits_oi = pd.DataFrame(
#     [
#         ["Multiple sclerosis"],
#         ["Type 1 diabetes"],
#         ["Inflammatory bowel disease"],
#         ["Crohn's disease"],
#         ["Systemic lupus erythematosus"],
#         ["Rheumatoid arthritis"],
#         ["Ankylosing spondylitis"],
#         ["Hodgkin's lymphoma"],
#         ["Psoriasis"],
#         ["Post bronchodilator FEV1/FVC ratio in COPD"],
#         ["Non-Hodgkin's lymphoma"],
#         ["Core binding factor acute myeloid leukemia"],
#         ["Chronic lymphocytic leukemia"],
#         ["Interleukin-6 levels"],
#         ["Interleukin-18 levels"],
#         [
#             "6-month creatinine clearance change response to tenofovir treatment in HIV infection (treatment arm interaction)"
#         ],
#         [
#             "Time-dependent creatinine clearance change response to tenofovir treatment in HIV infection (time and treatment arm interaction)"
#         ],
#         ["Autoimmune thyroid disease"],
#         ["IgG glycosylation"],
#         ["Asthma"],
#         ["Allergic disease (asthma, hay fever or eczema)"],
#         ["High IL-1beta levels in gingival crevicular fluid"],
#         ["C-reactive protein levels (MTAG)"],
#         ["Behcet's disease"],
#         ["Neutrophil count"],
#         ["Eosinophil counts"],
#         ["Monocyte count"],
#         ["Lymphocyte count"],
#         ["Endometriosis"],
#         ["Medication use (thyroid preparations)"],
#         ["Basophil count"],
#         [
#             "Acute graft versus host disease in bone marrow transplantation (recipient effect)"
#         ],
#         ["Selective IgA deficiency"],
#         ["Lymphocyte-to-monocyte ratio"],
#         ["COVID-19"],
#         ["C-reactive protein"],
#     ],
#     columns=["disease/trait"],
# ).set_index("disease/trait")
# motifscan_name = "gwas_immune"

# traits_oi = pd.DataFrame([
#     *trait_counts.loc[trait_counts.index.str.contains("erythro", flags = re.IGNORECASE)][:20].index,
#     *trait_counts.loc[trait_counts.index.str.contains("red blood", flags = re.IGNORECASE)][:20].index,
#     *trait_counts.loc[trait_counts.index.str.contains("hematocrit", flags = re.IGNORECASE)][:20].index,
#     *trait_counts.loc[trait_counts.index.str.contains("hemoglobin", flags = re.IGNORECASE)][:20].index,
#     *trait_counts.loc[trait_counts.index.str.contains("red cell", flags = re.IGNORECASE)][:20].index,
# ], columns = ["disease/trait"]).set_index("disease/trait"); motifscan_name = "gwas_hema"

traits_oi = pd.DataFrame(
    [
        ["Asthma"],
        ["Allergic disease (asthma, hay fever or eczema)"],
        ["Asthma (childhood onset)"],
        ["Atopic asthma"],
        ["Asthma (adult onset)"],
    ],
    columns=["disease/trait"],
).set_index("disease/trait")
motifscan_name = "gwas_asthma"

# %%
traits_oi.to_csv(folder_qtl / f"traits_{motifscan_name}.tsv", sep="\t", index=True) # store for other databases where we want to reuse the same traits, e.g. causaldb
traits_oi = traits_oi.loc[~traits_oi.index.duplicated()]
# manuscript.store_supplementary_table(traits_oi.reset_index(), f"traits_{motifscan_name}")

# %%
qtl = qtl.loc[qtl["disease/trait"].isin(traits_oi.index)]
qtl.shape

# %%
qtl["disease/trait"].value_counts()

# %%
qtl["snps_split"] = qtl["snps"].str.split("; ")
qtl = qtl.explode("snps_split").rename(columns={"snps_split": "snp"})

# %%
qtl = qtl.loc[qtl["snps"].str.startswith("rs")]

# %%
qtl.index = np.arange(len(qtl.index))

# %%
qtl_oi = qtl.sort_values("p-value", ascending=False).drop_duplicates(
    ["snp", "disease/trait"]
)[["disease/trait", "snp", "p-value", "strongest_risk_allele", "or or beta"]]

# %%
qtl_oi["strongest_risk_allele"].value_counts()

# %%
qtl_oi["rsid"] = qtl_oi["snp"]

# %% [markdown]
# ## Get LD

# %%
import requests

# %%
lddb_file = pathlib.Path(chd.get_output() / "lddb.pkl")
if not lddb_file.exists():
    lddb = {}
    pickle.dump(lddb, lddb_file.open("wb"))
lddb = pickle.load(lddb_file.open("rb"))
len(lddb)

# %%
np.sum([len(ld) + 1 for ld in lddb.values()])

# %%
# number of non-ld snps
np.mean([len(ld) == 0 for ld in lddb.values()])

# %%
sum([rsid not in lddb for rsid in qtl_oi["rsid"]])


# %%
def get_ld_data(rsid):
    # documentation at
    # https://rest.ensembl.org/documentation/info/ld_id_get
    import requests, sys

    server = "https://rest.ensembl.org"
    ext = f"/ld/human/{rsid}/1000GENOMES:phase_3:KHV?r2=0.9"

    r = requests.get(server + ext, headers={"Content-Type": "application/json"})

    if not r.ok:
        try:
            r.raise_for_status()
        except requests.HTTPError:
            return []
    else:
        decoded = r.json()
        return [
            {"snp1": var["variation1"], "snp2": var["variation2"], "r2": var["r2"]}
            for var in decoded
        ]


# %%
for rsid in tqdm.tqdm(qtl_oi["rsid"]):
    if rsid not in lddb:
        lddb[rsid] = get_ld_data(rsid)

# %%
pickle.dump(lddb, lddb_file.open("wb"))

# %%
len(lddb)

# %% [markdown]
# ### Combine SNPs in LD

# %%
import itertools

# %%
qtl_mapped = []
for _, qtl_info in qtl_oi.iterrows():
    qtl_mapped.append(
        {**qtl_info.to_dict(), "snp": qtl_info["snp"], "snp_main": qtl_info["snp"]}
    )
    for ld_info in lddb[qtl_info["snp"]]:
        qtl_mapped.append(
            {**qtl_info.to_dict(), "snp": ld_info["snp2"], "snp_main": qtl_info["snp"]}
        )
qtl_mapped = pd.DataFrame(qtl_mapped)

# %%
qtl_mapped = qtl_mapped.sort_values("p-value", ascending=False).drop_duplicates(
    ["snp", "disease/trait"]
)

# %%
qtl_mapped.to_pickle(folder_qtl / ("qtl_mapped_" + motifscan_name + ".pkl"))

# %% [markdown]
# ### Get SNP info

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

# %%
snp_info.to_pickle(folder_qtl / ("snp_info_" + motifscan_name + ".pkl"))

# %%

# %%
