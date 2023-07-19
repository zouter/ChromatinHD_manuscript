import chromatinhd as chd
from chromatinhd.models.diff.plot import Genes as GenesBase
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


def get_genes_plotdata(promoter, genome_folder, window):
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

    if len(result) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

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

    return plotdata_genes, plotdata_exons, plotdata_coding


class Genes(GenesBase):
    def __init__(self, promoter, genome_folder, window, *args, **kwargs):
        plotdata_genes, plotdata_exons, plotdata_coding = get_genes_plotdata(
            promoter, genome_folder, window
        )

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


from chromatinhd.grid.broken import Broken, Panel
from chromatinhd.plotting import gene_ticker
import matplotlib as mpl
import seaborn as sns


def filter_start_end(x, start, end):
    y = x.loc[~((x["end"] < start) | (x["start"] > end))]
    return y


def filter_position(x, start, end):
    y = x.loc[~((x["position"] < start) | (x["position"] > end))]
    return y


class GenesBrokenBase(Broken):
    def __init__(
        self,
        plotdata_genes,
        plotdata_exons,
        plotdata_coding,
        regions,
        gene_id,
        promoter,
        window,
        width,
        gap,
        full_ticks=False,
        *args,
        **kwargs,
    ):
        height = len(plotdata_genes) * 0.08
        super().__init__(
            regions=regions,
            height=height,
            width=width,
            gap=gap,
            *args,
            **kwargs,
        )

        ylim = (-0.5, plotdata_genes["ix"].max() + 0.5)

        for ((region, region_info), (panel, ax)) in zip(
            regions.iterrows(), self.elements[0]
        ):
            ax.xaxis.tick_top()
            ax.set_yticks([])
            ax.set_ylabel("")
            # ax.set_xlabel("Distance to TSS")
            ax.xaxis.set_label_position("top")
            ax.tick_params(axis="x", length=2, pad=0, labelsize=8, width=0.5)
            ax.xaxis.set_major_formatter(gene_ticker)

            for spine in ax.spines.values():
                spine.set_visible(False)

            ax.set_xlim(region_info["start"], region_info["end"])
            ax.set_ylim(*ylim)

            plotdata_genes_region = filter_start_end(
                plotdata_genes, region_info["start"], region_info["end"]
            )

            for gene, gene_info in plotdata_genes_region.iterrows():
                y = gene_info["ix"]
                is_oi = gene == gene_id
                ax.plot(
                    [gene_info["start"], gene_info["end"]],
                    [y, y],
                    color="black" if is_oi else "grey",
                )

                plotdata_exons_gene = plotdata_exons.query("gene == @gene")
                plotdata_exons_gene = filter_start_end(
                    plotdata_exons_gene, region_info["start"], region_info["end"]
                )
                h = 1
                for exon, exon_info in plotdata_exons_gene.iterrows():
                    rect = mpl.patches.Rectangle(
                        (exon_info["start"], y - h / 2),
                        exon_info["end"] - exon_info["start"],
                        h,
                        fc="white",
                        ec="#333333",
                        lw=1.0,
                        zorder=9,
                    )
                    ax.add_patch(rect)

                plotdata_coding_gene = plotdata_coding.query("gene == @gene")
                plotdata_coding_gene = filter_start_end(
                    plotdata_coding_gene, region_info["start"], region_info["end"]
                )
                for coding, coding_info in plotdata_coding_gene.iterrows():
                    rect = mpl.patches.Rectangle(
                        (coding_info["start"], y - h / 2),
                        coding_info["end"] - coding_info["start"],
                        h,
                        fc="#333333",
                        ec="#333333",
                        lw=1.0,
                        zorder=10,
                    )
                    ax.add_patch(rect)

            # vline at tss
            ax.axvline(0, color="#888888", lw=0.5, zorder=-1, dashes=(2, 2))

        ax = self[0, 0].ax
        ax.set_yticks(np.arange(len(plotdata_genes)))
        ax.set_yticklabels(plotdata_genes["symbol"], fontsize=6, style="italic")
        ax.tick_params(axis="y", length=0, pad=2, width=0.5)


class GenesBroken(GenesBrokenBase):
    def __init__(self, promoter, genome_folder, window, *args, **kwargs):
        plotdata_genes, plotdata_exons, plotdata_coding = get_genes_plotdata(
            promoter, genome_folder, window
        )

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
