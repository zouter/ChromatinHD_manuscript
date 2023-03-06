#%%
import gzip
import pandas as pd
import scanpy as sc
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

genes["tss"] = [genes_row["start"] if genes_row["strand"] == +1 else genes_row["end"] for _, genes_row in genes.iterrows()]
genes["window_end"] = [genes_row["tss"] - 200 if genes_row["strand"] == +1 else genes_row["end"] + 200 for _, genes_row in genes.iterrows()]

#%%
def extract_sequence(reference_file, chromosome, start, end):
    with gzip.open(reference_file, 'rt') as f:
        for record in SeqIO.parse(f, 'fasta'):
            # print(record.id)
            if record.id == chromosome:
                print('chrom found')
                sequence = str(record.seq[start:end]).upper()
                return sequence
    raise ValueError(f'Chromosome {chromosome} not found in file {reference_file}')

# %%
for index, row in genes.iterrows():
    start, end = sorted([row['tss'], row['window_end']])
    region = extract_sequence(file, row['chr'].replace('chr', ''), start, end)
    if row['strand'] == -1:
        region = region[::-1]
    genes.at[index, 'region'] = region
    print(index)

#%%
pattern1 = 'TATAA'
pattern2 = 'TATAAA'

genes['TATA'] = genes['region'].str.contains(pattern1) | genes['region'].str.contains(pattern2)

genes.to_csv(folder_data_preproc / "TATA.csv", index_col = 0)

#%%
