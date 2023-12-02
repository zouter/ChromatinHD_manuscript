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

import gzip
import torch

# %%
import chromatinhd as chd

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"
dataset_name = "pbmc10k"; organism = "hs"

promoter_name = "100k100k"

folder_data_preproc = folder_data / dataset_name

transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")

# %%
folder_qtl = chd.get_output() / "data" / "qtl" / "hs" / "gwas"
folder_qtl.mkdir(exist_ok = True, parents=True)

qtl_name = "gwas_immune"

# %%
qtlscan = chd.data.Motifscan(chd.get_output() / "motifscans" / dataset_name / promoter_name / qtl_name)

# %%
snps_info = pickle.load((chd.get_output() / "snp_info.pkl").open("rb"))
snps_info["strand_int"] = [+1 if strand == "+" else -1 for strand in snps_info["strand"]]
snps_info["alleles_list"] = snps_info["alleles"].str.decode("utf-8").str.split(",").str[:-1]

associations = pd.read_pickle(qtlscan.path / "association.pkl")

# %%
snps_info = snps_info.loc[associations["snp"].unique()]

# %%
# snps = pd.DataFrame({"chr":["chr1"], "pos":[100000], "strand":[1]})
# snps["start"] = snps["pos"] - 30
# snps["end"] = snps["pos"] + 30
# sequences = get_sequences(snps, genome)
# import torch

# %% [markdown]
# ### Load motifs

# %%
# motifscan_name = cutoff_col = "cutoff_0001"
motifscan_name = cutoff_col = "cutoff_001"

qtlchanges_folder = qtlscan.path / "changes" / motifscan_name
qtlchanges_folder.mkdir(parents = True, exist_ok = True)

# %%
folder_motifs = chd.get_output() / "data" / "motifs" / organism / "hocomoco"
folder_motifs.mkdir(parents=True, exist_ok=True)

# %%
pwms = pickle.load((folder_motifs / "pwms.pkl").open("rb"))
motifs = pd.read_pickle(folder_motifs / "motifs.pkl")

# motifs_oi = motifs.loc[motifs["gene_label"].isin(["TCF7", "GATA3", "IRF4"])]
# motifs_oi = motifs.iloc[:20]
# motifs_oi = motifs.loc[motifs["gene_label"].isin(["TCF7", ])]
motifs_oi = motifs

# %% [markdown]
# ### Create sequences (wild-type vs mutated)

# %%
genome = pickle.load(gzip.GzipFile((folder_data_preproc / "genome" / "genome.pkl.gz"), "rb"))


# %%
def create_onehot(seq):
    """
    Sequence contains integers 0 (A), 1 (C), 2 (G), 3 (T), and 4 (N)
    """
    return torch.tensor(np.eye(5, dtype=np.float32)[seq][:, :-1])
def get_sequence(promoter, genome):
    # have to add a +1 here because the genome annotation starts from 1 while python starts from 0
    sequence = genome[promoter["chr"]][promoter["start"] + 1 : promoter["end"] + 1]
    # flip sequence if strand is negative
    if promoter["strand_int"] == -1:
        sequence = sequence[::-1]
        sequence = np.array([3, 2, 1, 0, 4])[sequence]

    sequence = create_onehot(sequence)
    return sequence
translate_table = {"A":0, "C":1, "G":2, "T":3, "N":4} # alphabetic order
def translate(x):
    return np.array([translate_table[x] for x in x])
def translate_onehot(x):
    return create_onehot(translate(x))


# %%
def mutate(snp_info, sequence):
    # alleles = snp_info["observed"].split("/")
    alleles = snp_info["alleles_list"]
    if snp_info["locType"] == "exact":
        sequences = []
        for allele in alleles:
            if allele == "-":
                continue
            sequence2 = sequence.clone()
            allele_onehot = translate_onehot(allele)
            sequence2[[pad-1]] = allele_onehot
            sequences.append(sequence2)
    elif snp_info["locType"] == "between":
        sequences = []
        for allele in alleles:
            if allele == "-":
                sequence2 = sequence.clone()
            else:
                sequence2 = torch.cat([sequence[:(pad-1)], translate_onehot(allele), sequence[(pad-1):]], 0)
            sequences.append(sequence2)
    elif snp_info["locType"] == "range":
        sequences = []
        for allele in alleles:
            if allele == "-":
                sequence2 = torch.cat([sequence[:(pad-1)], sequence[(pad-1+len(snp_info["refUCSC"])):]], 0)#sequence.clone()
            else:
                sequence2 = sequence
            sequences.append(sequence2)
    elif snp_info["locType"] == "rangeSubstitution":
        sequences = []
        for allele in alleles:
            if allele == "-":
                sequence2 = sequence.clone()
                sequence2[(pad-1):(pad-1+len(snp_info["refUCSC"]))] = allele_onehot
            else:
                sequence2 = sequence
            sequences.append(sequence2)
    else:
        sequences = [sequence]
        
    max_size = max([sequence.shape[0] for sequence in sequences])
    sequences = [torch.nn.functional.pad(sequence, (0, 0, 0, max_size - sequence.shape[0]), value = torch.nan) for sequence in sequences]
    sequences = torch.stack(sequences)
    return sequences


# %%
bits_cutoff = np.log2(2)
def check_significant_changes(sequences, motifs_oi, pwms):
    significant_changes = []
    for motif in motifs_oi.index:
        cutoff = motifs_oi.loc[motif, cutoff_col]
        pwm = pwms[motif].to(sequences.device)
        score = scan(sequences, pwm)
        score = torch.nan_to_num(score, nan = -10.)
        
        found = (score > cutoff)
        
        scores = (score * found).sum(1)
        
        if (scores.max() > cutoff) and ((scores.max() - scores.min()) > bits_cutoff):
            significant_changes.append({
                "motif":motif,
                "tot_score":scores.max().item(),
                "best_score":score.max().item(),
                "diff_score":(scores.min() - scores.max()).item(),
                "diff_n":(found.sum(1).min() - found.sum(1).max()).item(),
                "best_n":(found.sum(1).max()).item()
            })
    significant_changes = pd.DataFrame(significant_changes)
    if len(significant_changes) == 0:
        significant_changes = pd.DataFrame(columns = ["motif"])
    return significant_changes
def find(sequences, motifs_oi, pwms):
    significant_changes = []
    for motif in motifs_oi.index:
        cutoff = motifs_oi.loc[motif, cutoff_col]
        pwm = pwms[motif].to(sequences.device)
        score = scan(sequences, pwm)
        score = torch.nan_to_num(score, nan = -10.)
        
        found = (score > cutoff)
        
        scores = (score * found).sum(1)
        
        if (scores.max() > cutoff) and ((scores.max() - scores.min()) > bits_cutoff):
            significant_changes.append({
                "motif":motif,
                "tot_score":scores.max().item(),
                "best_score":score.max().item(),
                "diff_score":(scores.min() - scores.max()).item(),
                "diff_n":(found.sum(1).min() - found.sum(1).max()).item(),
                "best_n":(found.sum(1).max()).item()
            })
    significant_changes = pd.DataFrame(significant_changes)
    if len(significant_changes) == 0:
        significant_changes = pd.DataFrame(columns = ["motif"])
    return significant_changes


# %%
def scan(onehot, pwm):
    n = onehot.shape[-2]
    k = pwm.shape[-2]

    # forward strand
    positive = torch.zeros(((*onehot.shape[:-2], n - k + 1)), device=onehot.device)
    for i in range(k):
        # to save memory we do the matrix multiplication once per motif position
        # this does not cause a significant slowdown
        x = torch.matmul(onehot, pwm[[i]].T)
        positive += x[..., i : n - k + i + 1, 0]
    del x

    # reverse (complement) strand
    onehot_comp = onehot[..., [3, 2, 1, 0]]
    pwm_rev = pwm.flip(0)
    negative = torch.zeros(((*onehot.shape[:-2], n - k + 1)), device=onehot.device)
    for i in range(k):
        x = torch.matmul(onehot_comp, pwm_rev[[i]].T)
        negative += x[..., i : n - k + i + 1, 0]
    del x

    # return maximum score across forward or reverse strands
    return torch.maximum(positive, negative)


# %%
changes_file = pathlib.Path(qtlchanges_folder / "changes.pkl")
if not changes_file.exists():
    changes = {}
    pickle.dump(changes, changes_file.open("wb"))
changes = pickle.load(changes_file.open("rb"))
len(changes)

# %%
# snps_info_oi = snps_info.iloc[:100]
# snps_info_oi = snps_info.iloc[[10]]
snps_info_oi = snps_info.loc[['rs8103453']]
# snps_info_oi = snps_info.query("locType == 'range'").head(1)
# snps_info_oi = snps_info.loc[snps_info.index.difference(pd.Index(changes.keys()))]

# %%
for snp, snp_info in tqdm.tqdm(snps_info_oi.iterrows(), total = len(snps_info_oi)):
    promoter = snp_info.copy()
    pad = 30
    promoter["start"] = int(promoter["start"] - pad)
    promoter["end"] = int(promoter["end"] + pad)
    sequence = get_sequence(promoter, genome)
    
    sequences = mutate(snp_info, sequence)
    
    changes_snp = check_significant_changes(sequences, motifs_oi, pwms).set_index("motif")
    changes[snp] = changes_snp

# %%
pickle.dump(changes, changes_file.open("wb"))

# %%
# for seq in sequences:
#     print("".join(np.array(["A", "C", "G", "T"])[seq.argmax(1)]))

# %% [markdown]
# ### Check

# %%
changes_file = pathlib.Path(qtlchanges_folder / "changes.pkl")

# %%
changes = pickle.load(changes_file.open("rb"))

# %%
changes_all = pd.concat(changes, names = ["snp"])

# %%
changes_all.loc["rs8103453"].sort_values("diff_score")

# %%
changes_all.loc[changes_all.index.get_level_values("motif").str.startswith("IRF")].sort_values("diff_score").head(10)

# %%
motifscan_name = cutoff_col

# %%
motifscan = chd.data.Motifscan(chd.get_output() / "motifscans" / dataset_name / promoter_name / motifscan_name)

# %%
motifs_oi["n_found"] = pd.Series(np.bincount(motifscan.indices, minlength = motifscan.n_motifs), index = motifscan.motifs.index)

# %%
motifs_oi["n_perturbed"] = changes_all.groupby("motif").size()

# %%
motifs_oi["found_over_perturbed"] = (motifs_oi["n_perturbed"] / motifs_oi["n_found"])

# %%
motifs_oi["found_over_perturbed"].sort_values()

# %%
fig, ax = plt.subplots()
motifs_oi["found_over_perturbed"].plot(kind = "hist")
motifs_oi.loc[motifs_oi.index.str.contains("IRF")]["found_over_perturbed"].plot(kind = "hist")

# %%
