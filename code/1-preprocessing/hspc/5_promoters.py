# %%
import pandas as pd
import chromatinhd as chd

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"
dataset_name = "hspc"
dataset_name_sub = "MV2"
folder_data_preproc = folder_data / dataset_name

# %%
genes = pd.read_csv(folder_data_preproc / "genes.csv", index_col = 0)
transcriptome = chd.data.Transcriptome(folder_data_preproc / f"{dataset_name_sub}_transcriptome")

# %%
promoter_name, (padding_negative, padding_positive) = "10k10k", (10000, 10000)

# %%
all_gene_ids = transcriptome.var['Accession']
all_gene_ids.index.name = "symbol"
all_gene_ids = all_gene_ids.reset_index()
all_gene_ids.index = all_gene_ids["Accession"]
all_gene_ids.index.name = "gene"

#%%
promoters = pd.DataFrame(index = all_gene_ids.index)
genes_missing = set(promoters.index).difference(set(genes.index))
genes_existing = set(promoters.index).intersection(set(genes.index))
promoters = promoters.loc[genes_existing]

# %%
promoters["tss"] = [genes_row["start"] if genes_row["strand"] == +1 else genes_row["end"] for _, genes_row in genes.loc[promoters.index].iterrows()]
promoters["strand"] = genes["strand"]
promoters["positive_strand"] = (promoters["strand"] == 1).astype(int)
promoters["negative_strand"] = (promoters["strand"] == -1).astype(int)
promoters["chr"] = genes.loc[promoters.index, "chr"]
promoters["start"] = promoters["tss"] - padding_negative * promoters["positive_strand"] - padding_positive * promoters["negative_strand"]
promoters["end"] = promoters["tss"] + padding_negative * promoters["negative_strand"] + padding_positive * promoters["positive_strand"]

# %%
promoters = promoters.drop(columns = ["positive_strand", "negative_strand"], errors = "ignore")
promoters.to_csv(folder_data_preproc / f"{dataset_name_sub}_promoters_{promoter_name}.csv")

# %%
