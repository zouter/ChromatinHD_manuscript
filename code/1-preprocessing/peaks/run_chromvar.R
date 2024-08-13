# if (!requireNamespace("BiocManager", quietly = TRUE))
#     install.packages("BiocManager")
# BiocManager::install("chromVAR")
# BiocManager::install("JASPAR2018")
# BiocManager::install("BSgenome.Hsapiens.UCSC.hg38")

library(chromVAR)
library(JASPAR2018)
library(Matrix)
library(dplyr)
library(tibble)
library(purrr)
library(future)
library(furrr)
library(SummarizedExperiment)

plan(multicore, workers = 24)

options(future.rng.onMisuse = "ignore")

args = commandArgs(trailingOnly=TRUE)

folder <- args[1]
# folder <- "./"
folder <- "/tmp/"

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

# split a vector of c("chr10:1000-10000", "chr1:10-500") into a dataframe of chr, start and end
split_chr <- function(x){
    x <- strsplit(x, ":")[[1]]
    chr <- x[1]
    x <- strsplit(x[2], "-")[[1]]
    start <- x[1]
    end <- x[2]
    data.frame(chr, start, end)
}
df <- map_dfr(rownames(counts), split_chr)
df$start <- as.numeric(df$start)
df$end <- as.numeric(df$end)
df$strand <- "+"
# add all columns in counts (which is a sparse matrix) to df
df2 <- cbind(df, as.matrix(counts)[, 1:ncol(counts)])
exp <- SummarizedExperiment::makeSummarizedExperimentFromDataFrame(df2)
assayNames(exp) <- "counts"

library(BSgenome.Hsapiens.UCSC.hg38)
exp <- addGCBias(exp, genome = BSgenome.Hsapiens.UCSC.hg38)

motif_ix <- seq(0, nrow(exp))

dev <- computeDeviations(object = exp)

diff_acc <- differentialDeviations(dev, "cluster")

# rename the default assay to count

pbmc$peak_region_fragments <-  colSums(pbmc@assays$peaks@counts)

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

