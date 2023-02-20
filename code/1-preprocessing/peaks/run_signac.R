# install.packages("readr")
# if (!require("BiocManager", quietly = TRUE))
#     install.packages("BiocManager")

# BiocManager::install("csaw")

# install.packages("tidyverse")

library(Signac)
library(Seurat)
library(Matrix)
library(dplyr)
library(tibble)
library(purrr)
library(future)
library(furrr)

plan(multicore, workers = 24)

options(future.rng.onMisuse = "ignore")

args = commandArgs(trailingOnly=TRUE)

folder <- args[1]
# folder <- "./"

print(folder)

counts <- t(Matrix::readMM(file.path(folder, "counts.mtx")))

var <- read.csv(file.path(folder, "var.csv"))
var$peak <- factor(var$peak)

obs <- read.csv(file.path(folder, "obs.csv"))
obs$cluster <- factor(obs$cluster)
obs$cell <- obs$peak
rownames(obs) <- obs$cell

clusters <- levels(obs$cluster)

 rownames(counts) <- var$peak
 colnames(counts) <- obs$peak

chrom_assay <- CreateAssayObject(
  counts = counts
)

pbmc <- CreateSeuratObject(
  counts = chrom_assay,
  assay = "peaks",
  meta.data = obs
)

DefaultAssay(pbmc) <- 'peaks'

pbmc$peak_region_fragments <-  colSums(pbmc@assays$peaks@counts)

# As described in the documentation, it is necessary to lower this min.pct 
# https://stuartlab.org/signac/articles/pbmc_vignette.html#find-differentially-accessible-peaks-between-clusters
# However, it is not described to what level
# Sadly, this is not even described in the original Signac paper...
# So we set it here to 0
# This really slows everything down though
# We also do not set the logfc threshold, because we will filter later anyway

# also, another bug is present in signac, which
# discussed in issues:
# - https://github.com/stuart-lab/signac/issues/1174
# - https://github.com/satijalab/seurat/issues/6915
# As of 2023-02-09, in Signac 1.9 and Seurat 4.3.0 this bug was not fixed
# The cutoff for log-fold change is adjusted accordingly in the python script

results <- furrr::future_map_dfr(clusters, function(cluster){
    set.seed(1)
# results <- map_dfr(clusters[1:2], function(cluster){
    da_peaks <- FindMarkers(
        object = pbmc,
        ident.1 = cluster,
        test.use = 'LR',
        latent.vars = 'peak_region_fragments',
        group.by = "cluster",
        min.pct = 0.000001,
        logfc.threshold = 0.
    )

    results <- rownames_to_column(da_peaks, "peak")
    results <- as_tibble(results)
    results$cluster <- cluster
    results
})

write.csv(results, file.path(folder, "results.csv"))

