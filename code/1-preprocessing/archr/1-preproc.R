devtools::install_github("GreenleafLab/ArchR", ref="master", repos = BiocManager::repositories())

library(ArchR)

addArchRGenome("hg38")
addArchRThreads(threads = 16) 

get_output <- function(){
  file.path(rprojroot::find_root(rprojroot::is_git_root), "output")
}

dataset_folder <- file.path(get_output(), "data", "pbmc10k")

inputFiles <- c(main = file.path(dataset_folder, "bam", "atac_fragments.tsv.gz"))

# ArrowFiles <- createArrowFiles(
#   inputFiles = inputFiles,
#   sampleNames = names(inputFiles),
#   minTSS = 4, #Dont set this too high because you can always increase later
#   minFrags = 1000, 
#   addTileMat = TRUE,
#   addGeneScoreMat = TRUE
# )

ArrowFiles <- file.path(get_output(), "data", "pbmc10k", "main.arrow")

archr_project_folder <- file.path(get_output(), "archr", "pbmc10k")
dir.create(archr_project_folder, recursive = TRUE)
proj <- ArchRProject(
  ArrowFiles = ArrowFiles, 
  outputDirectory = archr_project_folder,
  copyArrows = TRUE
)
saveArchRProject(ArchRProj = proj, outputDirectory = archr_project_folder, load = FALSE)

p1 <- plotFragmentSizes(ArchRProj = proj)


##

proj <- loadArchRProject(archr_project_folder)

library(TxDb.Hsapiens.UCSC.hg38.knownGene)
txdb <- TxDb.Hsapiens.UCSC.hg38.knownGene
chrom_sizes <- seqlengths(txdb)

promoters <- read_csv(file.path(dataset_folder, "promoters_100k100k.csv"))
promoters$start <- pmax(promoters$start, 100)
promoters$end <- pmin(promoters$end, chrom_sizes[as.character(seqnames(peaks))])
library(GenomicRanges)
promoters$strand <- as.character(list(`-1`="-", `1`="+")[as.character(promoters$strand)])
peaks <- makeGRangesFromDataFrame(promoters)

# BiocManager::install("TxDb.Hsapiens.UCSC.hg38.knownGene")


proj <- addPeakSet(proj, peaks, force = TRUE)
proj <- addMotifAnnotations(proj)

saveArchRProject(ArchRProj = proj, outputDirectory = archr_project_folder, load = FALSE)


motifPositions <- getPositions(proj)

# add celltypes
library(reticulate)
reticulate::use_condaenv("chromatinhd")
pandas <- reticulate::import("pandas")
celltype <- pandas$read_pickle(file.path(dataset_folder, "latent", "celltype.pkl"))
celltypes_external <- setNames(colnames(celltype)[apply(as.matrix(celltype), 1, which.max)], paste0("main#", rownames(celltype)))
celltypes <- setNames(rep("other", length(proj$cellNames)), proj$cellNames)
length(celltypes)
celltypes_external <- celltypes_external[names(celltypes_external) %in% names(celltypes)]
celltypes[names(celltypes_external)] <- celltypes_external

proj@cellColData$cluster <- celltypes

# get motifs of interest
motifs <- c("SPI1", "IRF3")
markerMotifs <- unlist(lapply(motifs, function(x) grep(x, names(motifPositions), value = TRUE)))
markerMotifs <- markerMotifs[markerMotifs %ni% "SREBF1_22"]
markerMotifs

# add coverage files to proj (as per https://www.archrproject.com/bookdown/motif-footprinting.html)
proj <- addGroupCoverages(ArchRProj = proj, groupBy = "cluster", force = TRUE)

# get positions (calculate and saved before)
motifPositions <- getPositions(proj)

# get footprints  (as per https://www.archrproject.com/bookdown/motif-footprinting.html)
seFoot <- getFootprints(
  ArchRProj = proj, 
  positions = motifPositions[markerMotifs], 
  groupBy = "cluster",
  nTop = 2000,
)

#
plots <- plotFootprints(seFoot, ArchRProj = proj)

# 
seFoot@assays@data$SPI1_322

#
positions_oi = list(a = motifPositions[markerMotifs]$SPI1_322[1], b = motifPositions[markerMotifs]$SPI1_322[1])

#
seFoot <- getFootprints(
  ArchRProj = proj, 
  positions = motifPositions[markerMotifs], 
  groupBy = "cluster",
  nTop = 2000,
)