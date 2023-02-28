#%%
import shutil
import requests
import itertools
import pandas as pd
import chromatinhd as chd

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"
dataset_name = "hspc"

folder_data_preproc = folder_data / dataset_name
folder_data_preproc.mkdir(exist_ok = True, parents = True)

mv1 = folder_data_preproc / 'MV1'
mv2 = folder_data_preproc / 'MV2'
mv1.mkdir(exist_ok = True, parents = True)
mv2.mkdir(exist_ok = True, parents = True)

#%%
files = {
    # 0 days, MV1
    'GSM6403408_3423-MV-1_gex_possorted_bam_0E7KE.loom.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6403nnn/GSM6403408/suppl/GSM6403408%5F3423%2DMV%2D1%5Fgex%5Fpossorted%5Fbam%5F0E7KE%2Eloom%2Egz',
    'GSM6403409_3423-MV-1_atac_fragments.tsv.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6403nnn/GSM6403409/suppl/GSM6403409%5F3423%2DMV%2D1%5Fatac%5Ffragments%2Etsv%2Egz',
    'GSM6403409_3423-MV-1_atac_fragments.tsv.gz.tbi.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6403nnn/GSM6403409/suppl/GSM6403409%5F3423%2DMV%2D1%5Fatac%5Ffragments%2Etsv%2Egz%2Etbi%2Egz',
    'GSM6403409_3423-MV-1_atac_peak_annotation.tsv.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6403nnn/GSM6403409/suppl/GSM6403409%5F3423%2DMV%2D1%5Fatac%5Fpeak%5Fannotation%2Etsv%2Egz',
    'GSE209878_3423-MV-1_barcodes.tsv.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE209nnn/GSE209878/suppl/GSE209878%5F3423%2DMV%2D1%5Fbarcodes%2Etsv%2Egz',
    'GSE209878_3423-MV-1_feature_linkage.bedpe.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE209nnn/GSE209878/suppl/GSE209878%5F3423%2DMV%2D1%5Ffeature%5Flinkage%2Ebedpe%2Egz',
    'GSE209878_3423-MV-1_features.tsv.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE209nnn/GSE209878/suppl/GSE209878%5F3423%2DMV%2D1%5Ffeatures%2Etsv%2Egz',
    'GSE209878_3423-MV-1_matrix.mtx.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE209nnn/GSE209878/suppl/GSE209878%5F3423%2DMV%2D1%5Fmatrix%2Emtx%2Egz',
    # 7 days, MV2
    'GSM6403410_3423-MV-2_gex_possorted_bam_ICXFB.loom.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6403nnn/GSM6403410/suppl/GSM6403410%5F3423%2DMV%2D2%5Fgex%5Fpossorted%5Fbam%5FICXFB%2Eloom%2Egz',
    'GSM6403411_3423-MV-2_atac_fragments.tsv.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6403nnn/GSM6403411/suppl/GSM6403411%5F3423%2DMV%2D2%5Fatac%5Ffragments%2Etsv%2Egz',    
    'GSM6403411_3423-MV-2_atac_fragments.tsv.gz.tbi.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6403nnn/GSM6403411/suppl/GSM6403411%5F3423%2DMV%2D2%5Fatac%5Ffragments%2Etsv%2Egz%2Etbi%2Egz',
    'GSM6403411_3423-MV-2_atac_peak_annotation.tsv.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6403nnn/GSM6403411/suppl/GSM6403411%5F3423%2DMV%2D2%5Fatac%5Fpeak%5Fannotation%2Etsv%2Egz',
    'GSE209878_3423-MV-2_barcodes.tsv.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE209nnn/GSE209878/suppl/GSE209878%5F3423%2DMV%2D2%5Fbarcodes%2Etsv%2Egz',
    'GSE209878_3423-MV-2_feature_linkage.bedpe.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE209nnn/GSE209878/suppl/GSE209878%5F3423%2DMV%2D2%5Ffeature%5Flinkage%2Ebedpe%2Egz',
    'GSE209878_3423-MV-2_features.tsv.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE209nnn/GSE209878/suppl/GSE209878%5F3423%2DMV%2D2%5Ffeatures%2Etsv%2Egz',
    'GSE209878_3423-MV-2_matrix.mtx.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE209nnn/GSE209878/suppl/GSE209878%5F3423%2DMV%2D2%5Fmatrix%2Emtx%2Egz'
}

for file in files:
    print(file)
    r = requests.get(files[file])
    with open(folder_data_preproc / file, 'wb') as f:
        f.write(r.content) 

#%%
shutil.move(folder_data_preproc/'GSE209878_3423-MV-1_barcodes.tsv.gz', mv1/'barcodes.tsv.gz')
shutil.move(folder_data_preproc/'GSE209878_3423-MV-1_features.tsv.gz', mv1/'features.tsv.gz')
shutil.move(folder_data_preproc/'GSE209878_3423-MV-1_matrix.mtx.gz', mv1/'matrix.mtx.gz')

shutil.move(folder_data_preproc/'GSE209878_3423-MV-2_barcodes.tsv.gz', mv2/'barcodes.tsv.gz')
shutil.move(folder_data_preproc/'GSE209878_3423-MV-2_features.tsv.gz', mv2/'features.tsv.gz')
shutil.move(folder_data_preproc/'GSE209878_3423-MV-2_matrix.mtx.gz', mv2/'matrix.mtx.gz')

#%%
!gzip -d {folder_data_preproc}/GSM6403408_3423-MV-1_gex_possorted_bam_0E7KE.loom.gz
!gzip -d {folder_data_preproc}/GSM6403410_3423-MV-2_gex_possorted_bam_ICXFB.loom.gz

!gzip -d {folder_data_preproc}/GSE209878_3423-MV-1_feature_linkage.bedpe.gz
!gzip -d {folder_data_preproc}/GSE209878_3423-MV-2_feature_linkage.bedpe.gz

!gzip -d {folder_data_preproc}/GSM6403409_3423-MV-1_atac_peak_annotation.tsv.gz
!gzip -d {folder_data_preproc}/GSM6403411_3423-MV-2_atac_peak_annotation.tsv.gz

!gzip -d {folder_data_preproc}/GSM6403409_3423-MV-1_atac_fragments.tsv.gz.tbi.gz
!gzip -d {folder_data_preproc}/GSM6403411_3423-MV-2_atac_fragments.tsv.gz.tbi.gz
# %% 
# Softlink relevant data
!ln -s {folder_data_preproc}/../brain/genes.gff.gz {folder_data_preproc}/genes.gff.gz
!ln -s {folder_data_preproc}/../brain//chromosome.sizes {folder_data_preproc}/chromosome.sizes
!ln -s {folder_data_preproc}/../brain/genome.pkl.gz {folder_data_preproc}/genome.pkl.gz
!ln -s {folder_data_preproc}/../brain/dna.fa.gz {folder_data_preproc}/dna.fa.gz
!ln -s {folder_data_preproc}/../brain/genes.csv {folder_data_preproc}/genes.csv

# %%
### https://doi.org/10.1126/science.aad0501
s_genes = [
    'MCM5', 'PCNA', 'TYMS', 'FEN1', 'MCM2', 'MCM4', 'RRM1', 'UNG', 'GINS2', 'MCM6', 
    'CDCA7', 'DTL', 'PRIM1', 'UHRF1', 'MLF1IP', 'HELLS', 'RFC2', 'RPA2', 'NASP', 
    'RAD51AP1', 'GMNN', 'WDR76', 'SLBP', 'CCNE2', 'UBR7', 'POLD3', 'MSH2', 'ATAD2', 
    'RAD51', 'RRM2', 'CDC45', 'CDC6', 'EXO1', 'TIPIN', 'DSCC1', 'BLM', 'CASP8AP2', 
    'USP1', 'CLSPN', 'POLA1', 'CHAF1B', 'BRIP1', 'E2F8'
]

g2m_genes = [
    'HMGB2', 'CDK1', 'NUSAP1', 'UBE2C', 'BIRC5', 'TPX2', 'TOP2A', 'NDC80', 
    'CKS2', 'NUF2', 'CKS1B', 'MKI67', 'TMPO', 'CENPF', 'TACC3', 'FAM64A', 
    'SMC4', 'CCNB2', 'CKAP2L', 'CKAP2', 'AURKB', 'BUB1', 'KIF11', 'ANP32E', 
    'TUBB4B', 'GTSE1', 'KIF20B', 'HJURP', 'HJURP', 'CDCA3', 'HN1', 'CDC20', 
    'TTK', 'CDC25C', 'KIF2C', 'RANGAP1', 'NCAPD2', 'DLGAP5', 'CDCA2', 
    'CDCA8', 'ECT2', 'KIF23', 'HMMR', 'AURKA', 'PSRC1', 'ANLN', 'LBR', 
    'CKAP5', 'CENPE', 'CTCF', 'NEK2', 'G2E3', 'GAS2L3', 'CBX5', 'CENPA'
]

### https://doi.org/10.1038/s41587-022-01476-y
hspc_marker_genes=[
    "SPINK2", "AZU1", "MPO", "ELANE", "TUBB1", "PF4", "PPBP", "LYZ", 
    "TCF4", "CD74", "HBB", "HBD", "KLF1", "PRG2", 'LMO4', 'EBF1'
]

lin_myeloid = ['HSC', 'MPP', 'LMPP', 'GMP']
lin_erythroid = ['HSC', 'MEP', 'Erythrocyte']
lin_platelet = ['HSC', 'MEP', 'Prog MK']

data = itertools.zip_longest(s_genes, g2m_genes, hspc_marker_genes, lin_myeloid, lin_erythroid, lin_platelet)
df = pd.DataFrame(data, columns=['s_genes', 'g2m_genes', 'hspc_marker_genes', 'lin_myeloid', 'lin_erythroid', 'lin_platelet'])

df.to_csv(folder_data_preproc / 'info_genes_cells.csv', index=False)

# %%
