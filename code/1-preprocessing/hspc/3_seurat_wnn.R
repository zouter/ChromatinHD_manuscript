#! /home/vifernan/projects/ChromatinHD_manuscript/software/R/R-4.2.2/bin/Rscript

# MultiVelo Seurat WNN Demo
# The procedure mostly follows Seurat tutorial: https://satijalab.org/seurat/articles/weighted_nearest_neighbor_analysis.html
# Note that we do not claim these preprocessing steps to be the best, as there aren't any. Feel free to make any changes you deem necessary.
# Please use libblas 3.9.1 and liblapack 3.9.1 for reproducing the 10X mouse brain demo, or use supplied WNN files on GitHub.

library(Seurat)
library(Signac)

getwd()

main_path = '/home/vifernan/projects/ChromatinHD_manuscript/output/data/hspc/'
dataset = 'MV2'

# read in expression and accessbility data
hspc.data <- Read10X(data.dir = paste0(main_path, dataset))

# subset for the same cells in the jointly filtered anndata object
barcodes <- read.delim(paste0(main_path, dataset, "_filtered_cells.txt"), header = F, stringsAsFactors = F)$V1

# preprocess RNA
hspc <- CreateSeuratObject(counts = hspc.data$`Gene Expression`[,barcodes])
hspc <- NormalizeData(hspc)
hspc <- FindVariableFeatures(hspc)
hspc <- ScaleData(hspc, do.scale = F) # not scaled for consistency with scVelo (optionally, use SCTransform)
hspc <- RunPCA(hspc, verbose = FALSE)
hspc <- RunUMAP(hspc, dims = 1:50, reduction.name = 'umap.rna', reduction.key = 'rnaUMAP_') # optional

# preprocess ATAC
hspc[["ATAC"]] <- CreateAssayObject(counts = hspc.data$`Peaks`[,barcodes], min.cells = 1)
DefaultAssay(hspc) <- "ATAC"
hspc <- RunTFIDF(hspc)
hspc <- FindTopFeatures(hspc, min.cutoff = 'q0')
hspc <- RunSVD(hspc)
hspc <- RunUMAP(hspc, reduction = 'lsi', dims = 2:50, reduction.name = "umap.atac", reduction.key = "atacUMAP_") # optional

# find weighted nearest neighbors
hspc <- FindMultiModalNeighbors(hspc, reduction.list = list("pca", "lsi"), dims.list = list(1:50, 2:50), k.nn = 50)
hspc <- RunUMAP(hspc, nn.name = "weighted.nn", reduction.name = "wnn.umap", reduction.key = "wnnUMAP_") # optional

# extract neighborhood graph
nn_idx <- hspc@neighbors$weighted.nn@nn.idx
nn_dist <- hspc@neighbors$weighted.nn@nn.dist
nn_cells <- hspc@neighbors$weighted.nn@cell.names

# save neighborhood graph
write.table(nn_idx, paste0(main_path, dataset, "_nn_idx.txt"), sep = ',', row.names = F, col.names = F, quote = F)
write.table(nn_dist, paste0(main_path, dataset, "_nn_dist.txt"), sep = ',', row.names = F, col.names = F, quote = F)
write.table(nn_cells, paste0(main_path, dataset, "_nn_cells.txt"), sep = ',', row.names = F, col.names = F, quote = F)

# save sessionInfo for reproducibility
writeLines(capture.output(sessionInfo()), paste0(main_path, dataset, "_sessionInfo.txt"))

print('3_seurat_wnn.R finished')