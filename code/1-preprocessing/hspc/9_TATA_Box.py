#%%
import math
import gzip
import torch
import pickle
import scipy.sparse
import numpy as np
import pandas as pd
import scanpy as sc
import tqdm.auto as tqdm
import chromatinhd as chd

from Bio import SeqIO

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"
dataset_name = "hspc"
folder_data_preproc = folder_data / dataset_name
file = folder_data_preproc / 'dna.fa.gz'

adata_result = sc.read_h5ad(folder_data_preproc / "multivelo_result.h5ad")

genes = pd.read_csv(folder_data_preproc / "genes.csv", index_col = 0)
genes = pd.merge(adata_result.var.loc[:, ['Accession']], genes, left_on='gene', right_on='symbol', how='left')

# !wget https://hocomoco11.autosome.org/final_bundle/hocomoco11/full/HUMAN/mono/pwm/TBP_HUMAN.H11MO.0.A.pwm -O {folder_data_preproc}/tatabox.txt

genes["tss"] = [genes_row["start"] if genes_row["strand"] == +1 else genes_row["end"] for _, genes_row in genes.iterrows()]
genes["window_end"] = [genes_row["tss"] - 200 if genes_row["strand"] == +1 else genes_row["end"] + 200 for _, genes_row in genes.iterrows()]

# %%
# genome = {}
# chromosomes = ["chr" + str(i) for i in range(24)] + ["chrX", "chrY"]
# chromosome = None

# translate_table = {"A":0, "C":1, "G":2, "T":3, "N":4} # alphabetic order
# for i, line in enumerate(gzip.GzipFile(folder_data_preproc / "dna.fa.gz")):
#     line = str(line,'utf-8')
#     if line.startswith(">"):
#         if chromosome is not None:
#             genome[chromosome] = np.array(genome_chromosome, dtype = np.int8)
#         chromosome = "chr" + line[1:line.find(" ")]
#         genome_chromosome = []
        
#         print(chromosome)
        
#         if chromosome not in chromosomes:
#             break
#     else:
#         genome_chromosome += [translate_table[x] for x in line.strip("\n").upper()]

# pickle.dump(genome, gzip.GzipFile((folder_data_preproc / "genome.pkl.gz"), "wb", compresslevel = 3))


#%%
genome = pickle.load(gzip.GzipFile((folder_data_preproc / "genome.pkl.gz"), "rb"))

promoter_name, window = "10k10k", np.array([-10000, 10000])
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0)

# %%
pwms = {}
motif = None
for line in (folder_data / "motifs/hs/hocomoco/pwms.txt").open():
    if line.startswith(">"):
        if motif is not None:
            pwms[motif_id] = motif
        motif_id = line[1:].strip("\n")
        motif = []
    else:
        motif.append([float(x) for x in line.split("\t")])

pwms = {motif_id: torch.tensor(pwm) for motif_id, pwm in pwms.items()}
pwms = {k: v for k, v in pwms.items() if k in ['TBP_HUMAN.H11MO.0.A']}

# %%
motifs = pd.DataFrame({"motif": pwms.keys()}).set_index("motif")
motifs["k"] = [pwms[motif].shape[0] for motif in motifs.index]

motif_cutoffs = pd.read_table(
    folder_data / "motifs/hs/hocomoco/pwm_cutoffs.txt",
    names=["motif", "cutoff_001", "cutoff_0005", "cutoff_0001"],
    skiprows=1,
).set_index("motif")

annot = (
    pd.read_table(folder_data / "motifs/hs/hocomoco/annot.txt")
    .rename(columns={"Model": "motif", "Transcription factor": "gene_label"})
    .set_index("motif")
)

motifs = motifs.join(motif_cutoffs)
motifs = motifs.join(annot)


#%%
# def extract_sequence(reference_file, chromosome, start, end):
#     with gzip.open(reference_file, 'rt') as f:
#         for record in SeqIO.parse(f, 'fasta'):
#             # print(record.id)
#             if record.id == chromosome:
#                 print('chrom found')
#                 sequence = str(record.seq[start:end]).upper()
#                 return sequence
#     raise ValueError(f'Chromosome {chromosome} not found in file {reference_file}')

# for index, row in genes.iterrows():
#     start, end = sorted([row['tss'], row['window_end']])
#     region = extract_sequence(file, row['chr'].replace('chr', ''), start, end)
#     if row['strand'] == -1:
#         region = region[::-1]
#     genes.at[index, 'region'] = region
#     print(index)

#%%
def create_onehot(seq):
    """
    Sequence contains integers 0 (A), 1 (C), 2 (G), 3 (T), and 4 (N)
    """
    return torch.tensor(np.eye(5, dtype=np.float32)[seq][:, :-1])

"""repeat with real TSS"""
def get_sequences(promoters, genome, window_length):
    onehot_promoters = torch.empty((promoters.shape[0], window_length, 4))
    for promoter_ix, (gene, promoter) in tqdm.tqdm(enumerate(promoters.iterrows()), total=promoters.shape[0]):
        # have to add a +1 here because the genome annotation starts from 1 while python starts from 0
        if promoter["strand"] == 1:
            start = promoter["start"] + 1 - window_length
            end = promoter["start"] + 1 
        else:
            start = promoter["end"] + 1
            end = promoter["end"] + 1 + window_length

        sequence = genome[promoter["chr"]][start : end]

        # flip sequence if strand is negative
        if promoter["strand"] == -1:
            sequence = sequence[::-1]
            sequence = np.array([3, 2, 1, 0, 4])[sequence]

        onehot_promoters[promoter_ix] = create_onehot(sequence)
    return onehot_promoters

onehot_promoters = get_sequences(promoters, genome, 50)

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

#%%
cutoff_col = "cutoff_001"
position_ixs = []
motif_ixs = []
scores = []

onehot_promoters = onehot_promoters.to("cuda")

for motif_ix, motif in enumerate(tqdm.tqdm(motifs.index)):
    cutoff = motifs.loc[motif, cutoff_col]
    pwm = pwms[motif].to(onehot_promoters.device)
    score = scan(onehot_promoters, pwm)
    pad = onehot_promoters.shape[-2] - score.shape[-1]
    pad_left = math.ceil(pad / 2)
    pad_right = math.floor(pad / 2)
    shape_left = (*score.shape[:-1], pad_left)
    shape_right = (*score.shape[:-1], pad_right)
    score = torch.cat(
        [
            torch.zeros(shape_left, device=score.device),
            score,
            torch.zeros(shape_right, device=score.device),
        ],
        dim=-1,
    )

    # add to sparse container
    # position in this case refers to gene x position (based on promoter window)
    position_ix = torch.where(score.flatten() > cutoff)[0].cpu().numpy()
    position_ixs.extend(position_ix)
    motif_ixs.extend(np.repeat(motif_ix, len(position_ix)))
    scores.extend(score.flatten().cpu().numpy()[position_ix])

onehot_promoters = onehot_promoters.to("cpu")

#%%
# convert to csr, but using coo as input
motifscores = scipy.sparse.csr_matrix(
    (scores, (position_ixs, motif_ixs)),
    shape=(np.prod(onehot_promoters.shape[:2]), motifs.shape[0]),
)





#%%
pattern1 = 'TATAA'
pattern2 = 'TATAAA'

genes['TATA'] = genes['region'].str.contains(pattern1) | genes['region'].str.contains(pattern2)

genes.to_csv(folder_data_preproc / "TATA.csv")

#%%
df = pd.read_csv(folder_data_preproc / "TATA.csv")
df['TATA'].sum() / df.shape[0]
# %%
