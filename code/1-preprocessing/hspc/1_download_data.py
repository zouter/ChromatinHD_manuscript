# %%
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
folder_data_preproc.mkdir(exist_ok=True, parents=True)

mv1 = folder_data_preproc / "MV1"
mv2 = folder_data_preproc / "MV2"
mv1.mkdir(exist_ok=True, parents=True)
mv2.mkdir(exist_ok=True, parents=True)

# %%
files = {
    # 0 days, MV1
    "GSM6403408_3423-MV-1_gex_possorted_bam_0E7KE.loom.gz": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6403nnn/GSM6403408/suppl/GSM6403408%5F3423%2DMV%2D1%5Fgex%5Fpossorted%5Fbam%5F0E7KE%2Eloom%2Egz",
    "GSM6403409_3423-MV-1_atac_fragments.tsv.gz": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6403nnn/GSM6403409/suppl/GSM6403409%5F3423%2DMV%2D1%5Fatac%5Ffragments%2Etsv%2Egz",
    "GSM6403409_3423-MV-1_atac_fragments.tsv.gz.tbi.gz": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6403nnn/GSM6403409/suppl/GSM6403409%5F3423%2DMV%2D1%5Fatac%5Ffragments%2Etsv%2Egz%2Etbi%2Egz",
    "GSM6403409_3423-MV-1_atac_peak_annotation.tsv.gz": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6403nnn/GSM6403409/suppl/GSM6403409%5F3423%2DMV%2D1%5Fatac%5Fpeak%5Fannotation%2Etsv%2Egz",
    "GSE209878_3423-MV-1_barcodes.tsv.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE209nnn/GSE209878/suppl/GSE209878%5F3423%2DMV%2D1%5Fbarcodes%2Etsv%2Egz",
    "GSE209878_3423-MV-1_feature_linkage.bedpe.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE209nnn/GSE209878/suppl/GSE209878%5F3423%2DMV%2D1%5Ffeature%5Flinkage%2Ebedpe%2Egz",
    "GSE209878_3423-MV-1_features.tsv.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE209nnn/GSE209878/suppl/GSE209878%5F3423%2DMV%2D1%5Ffeatures%2Etsv%2Egz",
    "GSE209878_3423-MV-1_matrix.mtx.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE209nnn/GSE209878/suppl/GSE209878%5F3423%2DMV%2D1%5Fmatrix%2Emtx%2Egz",
    # 7 days, MV2
    "GSM6403410_3423-MV-2_gex_possorted_bam_ICXFB.loom.gz": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6403nnn/GSM6403410/suppl/GSM6403410%5F3423%2DMV%2D2%5Fgex%5Fpossorted%5Fbam%5FICXFB%2Eloom%2Egz",
    "GSM6403411_3423-MV-2_atac_fragments.tsv.gz": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6403nnn/GSM6403411/suppl/GSM6403411%5F3423%2DMV%2D2%5Fatac%5Ffragments%2Etsv%2Egz",
    "GSM6403411_3423-MV-2_atac_fragments.tsv.gz.tbi.gz": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6403nnn/GSM6403411/suppl/GSM6403411%5F3423%2DMV%2D2%5Fatac%5Ffragments%2Etsv%2Egz%2Etbi%2Egz",
    "GSM6403411_3423-MV-2_atac_peak_annotation.tsv.gz": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6403nnn/GSM6403411/suppl/GSM6403411%5F3423%2DMV%2D2%5Fatac%5Fpeak%5Fannotation%2Etsv%2Egz",
    "GSE209878_3423-MV-2_barcodes.tsv.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE209nnn/GSE209878/suppl/GSE209878%5F3423%2DMV%2D2%5Fbarcodes%2Etsv%2Egz",
    "GSE209878_3423-MV-2_feature_linkage.bedpe.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE209nnn/GSE209878/suppl/GSE209878%5F3423%2DMV%2D2%5Ffeature%5Flinkage%2Ebedpe%2Egz",
    "GSE209878_3423-MV-2_features.tsv.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE209nnn/GSE209878/suppl/GSE209878%5F3423%2DMV%2D2%5Ffeatures%2Etsv%2Egz",
    "GSE209878_3423-MV-2_matrix.mtx.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE209nnn/GSE209878/suppl/GSE209878%5F3423%2DMV%2D2%5Fmatrix%2Emtx%2Egz",
}

for file in files:
    print(file)
    r = requests.get(files[file])
    with open(folder_data_preproc / file, "wb") as f:
        f.write(r.content)

# %%
shutil.move(folder_data_preproc / "GSE209878_3423-MV-1_barcodes.tsv.gz", mv1 / "barcodes.tsv.gz")
shutil.move(folder_data_preproc / "GSE209878_3423-MV-1_features.tsv.gz", mv1 / "features.tsv.gz")
shutil.move(folder_data_preproc / "GSE209878_3423-MV-1_matrix.mtx.gz", mv1 / "matrix.mtx.gz")

shutil.move(folder_data_preproc / "GSE209878_3423-MV-2_barcodes.tsv.gz", mv2 / "barcodes.tsv.gz")
shutil.move(folder_data_preproc / "GSE209878_3423-MV-2_features.tsv.gz", mv2 / "features.tsv.gz")
shutil.move(folder_data_preproc / "GSE209878_3423-MV-2_matrix.mtx.gz", mv2 / "matrix.mtx.gz")

# %%
!mv {folder_data_preproc}/GSM6403411_3423-MV-2_atac_fragments.tsv.gz {folder_data_preproc}/atac_fragments.tsv.gz
!mv {folder_data_preproc}/GSM6403411_3423-MV-2_atac_fragments.tsv.gz.tbi {folder_data_preproc}/atac_fragments.tsv.gz.tbi

# %%
ln -s MV2/atac_fragments.tsv.gz ./atac_fragments.tsv.gz
ln -s MV2/atac_fragments.tsv.gz.tbi ./atac_fragments.tsv.gz.tbi

# %%
peak_annot = pd.read_table(folder_data_preproc / "GSM6403411_3423-MV-2_atac_peak_annotation.tsv.gz")
peaks = peak_annot[["chrom", "start", "end"]]
peaks_folder = chd.get_output() / "peaks" / dataset_name / "cellranger"
peaks_folder.mkdir(exist_ok=True, parents=True)
peaks.to_csv(peaks_folder / "peaks.bed", sep="\t", index=False, header=False)