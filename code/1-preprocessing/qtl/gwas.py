# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns
sns.set_style('ticks')

import pickle

import tqdm.auto as tqdm

import pathlib

import polars as pl

# %%
import chromatinhd as chd

# %%
folder_qtl = chd.get_output() / "data" / "qtl" / "hs" / "gwas"
folder_qtl.mkdir(exist_ok = True, parents=True)

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
# # !wget http://www.ebi.ac.uk/efo/efo.owl

# %%
import owlready2

# %%
onto = owlready2.get_ontology("efo.owl").load()

# %%
immune_system_disease = onto.search(iri = "http://www.ebi.ac.uk/efo/EFO_0000540")[0]

# %%
children = onto.search(subclass_of = immune_system_disease)

# %%
rdf = onto.get_namespace("http://www.w3.org/2000/01/rdf-schema#")
obo = onto.get_namespace("http://purl.obolibrary.org/obo/")

# %%
traits = []
for child in children:
    description = obo.IAO_0000115[child][0] if len(obo.IAO_0000115[child]) else ""
    traits.append({"trait":rdf.label[child][0], "description":description})
traits = pd.DataFrame(traits)

# %% [markdown]
# ## Process

# %%
qtl = pd.read_table(folder_qtl/"full.tsv", index_col = None)
qtl = qtl.rename(columns = dict(zip(qtl.columns, [col.lower() for col in qtl.columns])))

# %%
qtl["strongest_risk_allele"] = qtl["strongest snp-risk allele"].str.split("-").str[-1]

# %%
diseases = qtl.groupby("disease/trait").size().sort_values(ascending = False)

# %%
trait_counts = qtl["disease/trait"].value_counts()

# %%
trait_counts.loc[trait_counts.index.str.contains("lymph")]

# %%
trait_counts.head(250).iloc[200:250]

# %%
# traits_oi = pd.DataFrame([
#     ["Attention deficit hyperactivity disorder or autism spectrum disorder or intelligence (pleiotropy)"],
#     ["Alzheimer’s disease polygenic risk score (upper quantile vs lower quantile)"],
# ], columns = ["disease/trait"]).set_index("disease/trait"); motifscan_name = "gwas_cns"

# traits_oi = pd.DataFrame([
#     ["Chronic lymphocytic leukemia"],
#     ["Acute lymphoblastic leukemia (childhood)"],
#     ["Hodgkin's lymphoma"],
#     ["Childhood ALL/LBL (acute lymphoblastic leukemia/lymphoblastic lymphoma) treatment-related venous thromboembolism"],
#     ["B-cell malignancies (chronic lymphocytic leukemia, Hodgkin lymphoma or multiple myeloma) (pleiotropy)"],
#     ["Non-Hodgkin's lymphoma"],
# ], columns = ["disease/trait"]).set_index("disease/trait"); motifscan_name = "gwas_lymphoma"

traits_oi = pd.DataFrame([
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
    ["6-month creatinine clearance change response to tenofovir treatment in HIV infection (treatment arm interaction)"],
    ["Time-dependent creatinine clearance change response to tenofovir treatment in HIV infection (time and treatment arm interaction)"],
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
    ["Acute graft versus host disease in bone marrow transplantation (recipient effect)"],
    ["Selective IgA deficiency"],
    ["Lymphocyte-to-monocyte ratio"],
    ["COVID-19"],
    ["C-reactive protein"],
], columns = ["disease/trait"]).set_index("disease/trait"); motifscan_name = "gwas_immune"
# traits_oi

# %%
qtl = qtl.loc[qtl["disease/trait"].isin(traits_oi.index)]
qtl

# %%
qtl["snps_split"] = qtl["snps"].str.split("; ")
qtl = qtl.explode("snps_split").rename(columns = {"snps_split":"snp"})

# %%
qtl = qtl.loc[qtl["snps"].str.startswith("rs")]

# %%
qtl.index = np.arange(len(qtl.index))

# %%
qtl_oi = qtl.sort_values('p-value', ascending=False).drop_duplicates(["snp", "disease/trait"])[["disease/trait", "snp", "p-value", "strongest_risk_allele", "or or beta"]]

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
sum([rsid not in lddb for rsid in qtl_oi["rsid"]])


# %%
def get_ld_data(rsid):
    # documentation at
    # https://rest.ensembl.org/documentation/info/ld_id_get
    import requests, sys

    server = "https://rest.ensembl.org"
    ext = f"/ld/human/{rsid}/1000GENOMES:phase_3:KHV?r2=0.9"

    r = requests.get(server+ext, headers={ "Content-Type" : "application/json"})

    if not r.ok:
        try:
            r.raise_for_status()
        except requests.HTTPError:
            return []
    else: 
        decoded = r.json()
        return [{"snp1":var["variation1"], "snp2":var["variation2"], "r2":var["r2"]} for var in decoded]


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
    qtl_mapped.append({**qtl_info.to_dict(), "snp":qtl_info["snp"], "snp_main":qtl_info["snp"]})
    for ld_info in lddb[qtl_info["snp"]]:
        qtl_mapped.append({**qtl_info.to_dict(), "snp":ld_info["snp2"], "snp_main":qtl_info["snp"]})
qtl_mapped = pd.DataFrame(qtl_mapped)

# %%
qtl_mapped = qtl_mapped.sort_values('p-value', ascending=False).drop_duplicates(["snp", "disease/trait"])

# %%
qtl_mapped.to_pickle(folder_qtl / ("qtl_mapped_" + motifscan_name + ".pkl"))

# %% [markdown]
# ### Get SNP info

# %%
snp_info_file = pathlib.Path(chd.get_output() / "snp_info.pkl")
if not snp_info_file.exists():
    snp_info = pd.DataFrame({"snp":pd.Series(dtype = str), "chr":pd.Series(dtype = str), "start":pd.Series(dtype = int), "end":pd.Series(dtype = int)}).set_index("snp")
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
chunks = [snps_missing[i:i+n] for i in range(0, len(snps_missing), n)]

# %%
len(chunks)

# %%
for snps in tqdm.tqdm(chunks):
    snp_names = ",".join("'" + snps + "'")
    query = f"select * from snp151 where name in ({snp_names})"
    result = pd.read_sql(query,"mysql+pymysql://genome@genome-mysql.cse.ucsc.edu/{organism}?charset=utf8mb4".format(organism='hg38')).set_index("name")
    result  = result.rename(columns = {"chrom":"chr", "chromStart":"start", "chromENd":"end", "name":"snp"})
    
    result["start"] = result["start"].astype(int)
    result["end"] = result["start"].astype(int)
    result.index.name = "snp"
    result = result.groupby("snp").first()
    result = result.reindex(snps)
    result.index.name = "snp"
    
    # assert result.index.str.contains(";").any()
    
    snp_info = pd.concat([snp_info, result], axis = 0)
    pickle.dump(snp_info, snp_info_file.open("wb"))

# %%
snp_info.to_pickle(folder_qtl / ("snp_info_" + motifscan_name + ".pkl"))

# %% [markdown]
# ## Create 

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

# dataset_name = "lymphoma"
dataset_name = "pbmc10k"
# dataset_name = "e18brain"
# dataset_name = "alzheimer"
# dataset_name = "brain"

# dataset_name = "FLI1_7"
# dataset_name = "PAX2_7"
# dataset_name = "NHLH1_7"
# dataset_name = "CDX2_7"
# dataset_name = "CDX1_7"
# dataset_name = "MSGN1_7"
# dataset_name = "KLF4_7"
# dataset_name = "KLF5_7"
# dataset_name = "PTF1A_4"

folder_data_preproc = folder_data / dataset_name

transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")

# %%
promoter_name, window = "10k10k", np.array([-10000, 10000])
# promoter_name, window = "1k1k", np.array([-1000, 1000])
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)

# %% [markdown]
# ### Link QTLs to SNP location

# %%
chromosomes = promoters["chr"].unique()

# %%
snp_info = pickle.load((chd.get_output() / "snp_info.pkl").open("rb"))
qtl_mapped = pd.read_pickle(folder_qtl / ("qtl_mapped_" + motifscan_name + ".pkl"))
qtl_mapped.index = np.arange(len(qtl_mapped))
association = qtl_mapped.join(snp_info, on = "snp")
association = association.loc[~pd.isnull(association["start"])]
association["pos"] = association["start"].astype(int)

# %%
import pybedtools

# %%
association_bed = pybedtools.BedTool.from_dataframe(association.reset_index()[["chr", "pos", "pos", "index"]])
promoters_bed = pybedtools.BedTool.from_dataframe(promoters[["chr", "start", "end"]])
intersection = association_bed.intersect(promoters_bed)
association = association.loc[intersection.to_dataframe()["name"].unique()]

# %%
chromosome_mapping = pd.Series(np.arange(len(chromosomes)), chromosomes)
promoters["chr_int"] = chromosome_mapping[promoters["chr"]].values

# %%
association = association.loc[association.chr.isin(chromosomes)].copy()

# %%
association["chr_int"] = chromosome_mapping[association["chr"]].values

# %%
association = association.sort_values(["chr_int", "pos"])

# %%
assert np.all(np.diff(association["chr_int"].to_numpy()) >= 0), "Should be sorted by chr"

# %%
motif_col = "disease/trait"

# %%
association[motif_col] = association[motif_col].astype("category")

# %%
len(association["snp"].unique())

# %%
assert association[motif_col].dtype.name == "category"

# %%
n = []

position_ixs = []
motif_ixs = []
scores = []

for gene_ix, promoter_info in enumerate(promoters.itertuples()):
    chr_int = promoter_info.chr_int
    chr_start = np.searchsorted(association["chr_int"].to_numpy(), chr_int)
    chr_end = np.searchsorted(association["chr_int"].to_numpy(), chr_int + 1)
    
    pos_start = chr_start + np.searchsorted(association["pos"].iloc[chr_start:chr_end].to_numpy(), promoter_info.start)
    pos_end = chr_start + np.searchsorted(association["pos"].iloc[chr_start:chr_end].to_numpy(), promoter_info.end)
    
    qtls_promoter = association.iloc[pos_start:pos_end].copy()
    qtls_promoter["relpos"] = qtls_promoter["pos"] - promoter_info.start
    
    if promoter_info.strand == -1:
        qtls_promoter = qtls_promoter.iloc[::-1].copy()
        qtls_promoter["relpos"] = -qtls_promoter["relpos"] + (window[1] - window[0]) + 1
        
    # if promoter_info.chr == 'chr6':
    #     qtls_promoter = qtls_promoter.loc[[]]
    
    n.append(len(qtls_promoter))
    
    position_ixs += (qtls_promoter["relpos"] + (gene_ix * (window[1] - window[0]))).astype(int).tolist()
    motif_ixs += (qtls_promoter[motif_col].cat.codes.values).astype(int).tolist()
    scores += ([1] * len(qtls_promoter))

# %% [markdown]
# Control with sequence

# %%
# onehot_promoters = pickle.load((folder_data_preproc / ("onehot_promoters_" + promoter_name + ".pkl")).open("rb"))
# qtls_promoter.groupby("snp").first().head(20)
# onehot_promoters[gene_ix, 11000]

# %%
promoters["n"] = n

# %%
(promoters["n"] == 0).mean()

# %%
promoters.sort_values("n", ascending = False).head(30).assign(symbol = lambda x:transcriptome.symbol(x.index).values)

# %%
promoters.sort_values("n", ascending = False).assign(symbol = lambda x:transcriptome.symbol(x.index).values).set_index("symbol").loc["POU2AF1"]

# %%
motifs_oi = association[[motif_col]].groupby([motif_col]).first()
motifs_oi["n"] = association.groupby(motif_col).size()

# %%
motifs_oi.sort_values("n", ascending = False)

# %%
import scipy.sparse

# convert to csr, but using coo as input
motifscores = scipy.sparse.csr_matrix((scores, (position_ixs, motif_ixs)), shape = (len(promoters) * (window[1] - window[0]), motifs_oi.shape[0]))

# %% [markdown]
# ### Save

# %%
import chromatinhd as chd

# %%
motifscan = chd.data.Motifscan(chd.get_output() / "motifscans" / dataset_name / promoter_name / motifscan_name)

# %%
association.to_pickle(motifscan.path / "association.pkl")

# %%
motifscan.indices = motifscores.indices
motifscan.indptr = motifscores.indptr
motifscan.data = motifscores.data
motifscan.shape = motifscores.shape

# %%
motifscan

# %%
pickle.dump(motifs_oi, open(motifscan.path / "motifs.pkl", "wb"))

# %%
motifscan.n_motifs = len(motifs_oi)

# %% [markdown]
# ### Plot qtls in promoter

# %%
promoter = pd.Series(
{'tss': 102777449,
 'strand': 1,
 'chr': 'chr14',
 'start': 102767449,
 'end': 102787449,
 'n': 33}
)

# %%
association_oi = association.query("chr == @promoter.chr").query("start > @promoter.start").query("end < @promoter.end")
association_oi = association_oi.query("`disease/trait` == 'Systemic lupus erythematosus'").copy()
association_oi["relpos"] = (association_oi["pos"] - promoter["tss"]) * (promoter["strand"])

# %%
fig, ax = plt.subplots()
for _, q in association_oi.iterrows():
    ax.axvline(q["relpos"])
ax.set_xlim(association_oi["relpos"].min(), association_oi["relpos"].max())

# %%
pos_oi = -10000
association_oi.assign(dist = lambda x:abs(x.relpos - pos_oi)).sort_values("dist").head(10)

# %%
# !ls -lh {motifscan.path}

# %%
snp_info_oi = snp_info.loc[qtl_mapped["snp"].unique()]

# %%
