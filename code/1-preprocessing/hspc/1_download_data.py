#%%
import shutil
import requests
import itertools
import subprocess
import pandas as pd
import chromatinhd as chd
from tqdm import tqdm

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

for file in tqdm(files.keys()):  # Iterate over the file keys and use tqdm to display the progress bar
    print(file)
    # check if file exists
    if not (mv1 / file).exists() and not (mv2 / file).exists() and not (folder_data_preproc / file).exists():
        print('Downloading file')
        response = requests.get(files[file], stream=True)  # Enable streaming for large files
        total_size = int(response.headers.get("content-length", 0))  # Get the total file size from the response headers
        progress_bar = tqdm(total=total_size, unit="B", unit_scale=True)  # Initialize the progress bar

        with open(folder_data_preproc / file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):  # Iterate over the response content in chunks
                if chunk:  # Filter out keep-alive new chunks
                    f.write(chunk)
                    progress_bar.update(len(chunk))  # Update the progress bar with the chunk size

        progress_bar.close()  # Close the progress bar after the download is complete

print('Downloaded all files')

#%%
for x in zip(['MV-1', 'MV-2'], [mv1, mv2]):
    print(x)
    # get all filenames for MV-1 or MV-2
    keys = [key for key in files if x[0] in key]
    print(keys)
    for key in keys:
        print(folder_data_preproc/key, x[1]/key)
        # move each file to the correct folder
        shutil.move(folder_data_preproc/key, x[1]/key)
        # unzip specific files
        endings = ['.loom.gz', '_feature_linkage.bedpe.gz', '_atac_peak_annotation.tsv.gz', '_atac_fragments.tsv.gz.tbi.gz']
        for ending in endings:
            if key.endswith(ending):
                full_path = f'{x[1]}/{key}'
                subprocess.run(['gzip', '-d', full_path])
                break
    
    # rename
    shutil.move(x[1]/ f'GSE209878_3423-{x[0]}_barcodes.tsv.gz', x[1]/'barcodes.tsv.gz')
    shutil.move(x[1]/ f'GSE209878_3423-{x[0]}_features.tsv.gz', x[1]/'features.tsv.gz')
    shutil.move(x[1]/ f'GSE209878_3423-{x[0]}_matrix.mtx.gz', x[1]/'matrix.mtx.gz')


#%%
# Softlink relevant data
subprocess.run(['ln', '-s', f'{folder_data_preproc}/../brain/genes.gff.gz', f'{folder_data_preproc}/genes.gff.gz'])
subprocess.run(['ln', '-s', f'{folder_data_preproc}/../brain//chromosome.sizes', f'{folder_data_preproc}/chromosome.sizes'])
subprocess.run(['ln', '-s', f'{folder_data_preproc}/../brain/genome.pkl.gz', f'{folder_data_preproc}/genome.pkl.gz'])
subprocess.run(['ln', '-s', f'{folder_data_preproc}/../brain/dna.fa.gz', f'{folder_data_preproc}/dna.fa.gz'])
subprocess.run(['ln', '-s', f'{folder_data_preproc}/../brain/genes.csv', f'{folder_data_preproc}/genes.csv'])

#%%
ref ={
    'dna.fa.gz': 'http://ftp.ensembl.org/pub/release-107/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna_sm.toplevel.fa.gz'
}

# for file in ref:
#     print(file)
#     r = requests.get(ref[file])
#     with open(folder_data_preproc / file, 'wb') as f:
#         f.write(r.content) 

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

print('1_download_data.py finished')

# %%
