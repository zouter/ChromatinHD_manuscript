import chromatinhd as chd
from chromatinhd.differential.plot import Genes as GenesBase
import pandas as pd
import pybedtools
import numpy as np


def center(coords, promoter, window):
    coords = coords.copy()
    if promoter.strand == 1:
        coords["start"] = coords["start"] - promoter["start"] + window[0]
        coords["end"] = coords["end"] - promoter["start"] + window[0]
    else:
        coords["start"] = (
            (window[1] - window[0]) - (coords["start"] - promoter["start"]) + window[0]
        )
        coords["end"] = (
            (window[1] - window[0]) - (coords["end"] - promoter["start"]) + window[0]
        )

        coords = coords.rename(columns={"start": "end", "end": "start"})

    return coords


class Genes(GenesBase):
    def __init__(self, promoter, genome_folder, window, *args, **kwargs):
        genes = pd.read_csv(genome_folder / "genes.csv", index_col=0)
        genes = genes.rename(columns={"Strand": "strand"})

        plotdata_genes = center(
            genes.loc[genes["chr"] == promoter["chr"]]
            .query("~((start > @promoter.end) | (end < @promoter.start))")
            .copy(),
            promoter,
            window,
        )
        plotdata_genes["ix"] = np.arange(len(plotdata_genes))

        gene_ids = plotdata_genes.index

        query = f"""<?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE Query>
        <Query  virtualSchemaName = "default" formatter = "TSV" header = "1" uniqueRows = "0" count = "" datasetConfigVersion = "0.6" >
                    
            <Dataset name = "hsapiens_gene_ensembl" interface = "default" >
                <Filter name = "ensembl_gene_id" value = "{','.join(gene_ids)}"/>
                <Filter name = "transcript_is_canonical" excluded = "0"/>

                <Attribute name = "ensembl_gene_id" />
                <Attribute name = "ensembl_gene_id_version" />
                <Attribute name = "exon_chrom_start" />
                <Attribute name = "exon_chrom_end" />
                <Attribute name = "genomic_coding_start" />
                <Attribute name = "genomic_coding_end" />
                <Attribute name = "ensembl_transcript_id" />
                <Attribute name = "ensembl_transcript_id_version" />
            </Dataset>
        </Query>"""
        result = chd.utils.biomart.get(query)

        # result = result.dropna().copy()
        plotdata_exons = (
            result[["Gene stable ID", "Exon region start (bp)", "Exon region end (bp)"]]
            .rename(
                columns={
                    "Gene stable ID": "gene",
                    "Exon region start (bp)": "start",
                    "Exon region end (bp)": "end",
                }
            )
            .dropna()
        )
        plotdata_exons = center(plotdata_exons, promoter, window)

        plotdata_coding = result.rename(
            columns={
                "Gene stable ID": "gene",
                "Genomic coding start": "start",
                "Genomic coding end": "end",
            }
        ).dropna()
        plotdata_coding = center(plotdata_coding, promoter, window)

        return super().__init__(
            *args,
            plotdata_genes=plotdata_genes,
            plotdata_exons=plotdata_exons,
            plotdata_coding=plotdata_coding,
            promoter=promoter,
            gene_id=promoter.name,
            window=window,
            **kwargs,
        )
