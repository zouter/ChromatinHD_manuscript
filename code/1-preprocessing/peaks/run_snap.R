library(dplyr)
library(edgeR)

tmp_folder <- "/tmp"

x <- read.csv(file.path(tmp_folder, "counts_cluster.csv"), stringsAsFactors=FALSE, header=TRUE, row.names=1)
y <- as.matrix(x)

bcv <- 0.1

tb.poss <- list()

# taken from snap atac
for (cluster in rownames(x)){
    print(cluster)
    y_cluster <- y[cluster,]
    y_notcluster <- colSums(y[rownames(y) != cluster,])

    data.use = data.frame(y_cluster, y_notcluster)
    group <- factor(c(1,2));
    design <- model.matrix(~group);
    dge <- DGEList(counts=data.use, group=group);

    tb.pos <- exactTest(dge, dispersion=bcv^2)$table;
    tb.pos$peak_ix <- rownames(tb.pos)

    tb.pos$cluster <- cluster

    tb.poss[[cluster]] <- tb.pos
}

tb.pos <- bind_rows(tb.poss)
write.csv(tb.pos, "/tmp/tb_pos.csv")