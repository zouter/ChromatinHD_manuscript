import pandas as pd
import numpy as np

import chromatinhd as chd
import chromatinhd.peakcounts

import pickle

import torch
import scanpy as sc
import tqdm.auto as tqdm

device = "cuda:1"

folder_root = chd.get_output()
folder_data = folder_root / "data"

import itertools

from designs

design = pd.DataFrame.from_records(
    itertools.chain(
        itertools.product(
            [
                # "pbmc10k",
                # "lymphoma",
                # "e18brain",
                # "brain",
                # "alzheimer",
                # "pbmc10k_gran"
                # "pbmc3k-pbmc10k",
                # "pbmc3k-pbmc10k",
                # "pbmc3k-pbmc10k",
            ],
            [
                "cellranger",
                "macs2",
                "rolling_50",
                "rolling_100",
                "rolling_500",
                "macs2_improved",
                # "genrich",
                "macs2_leiden_0.1_merged",
                "macs2_leiden_0.1",
                "stack",
                "encode_screen",
            ],
        )
    ),
    columns=["dataset", "peakcaller"],
)
# design = design.loc[
#     ~((design["dataset"] == "alzheimer") & (design["peakcaller"] == "genrich"))
# ]
design["force"] = False
print(design)

for dataset_name, design_dataset in design.groupby("dataset"):
    print(f"{dataset_name=}")
    # transcriptome
    folder_data_preproc = folder_data / dataset_name

    # fragments
    # promoter_name, window = "1k1k", np.array([-1000, 1000])
    promoter_name, window = "10k10k", np.array([-10000, 10000])
    # promoter_name, window = "20kpromoter", np.array([-10000, 0])
    promoters = pd.read_csv(
        folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
    )
    window_width = window[1] - window[0]

    fragments = chd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)
    fragments.window = window

    print(design_dataset)
    for peakcaller, design_peaks in design_dataset.groupby("peakcaller"):
        print(peakcaller)
        peakcounts = chd.peakcounts.FullPeak(
            folder=chd.get_output() / "peakcounts" / dataset_name / peakcaller
        )

        design_row = design_peaks.iloc[0]

        force = design_row["force"]
        try:
            peakcounts.counts
        except BaseException:
            force = True or force

        if force:
            if peakcaller == "stack":
                peaks = promoters.reset_index()[["chr", "start", "end", "gene"]]
            elif peakcaller.startswith("rolling"):
                step_size = int(peakcaller.split("_")[1])
                peaks = []
                for gene, promoter in promoters.iterrows():
                    starts = np.arange(
                        promoter["start"], promoter["end"], step=step_size
                    )
                    ends = np.hstack([starts[1:], [promoter["end"]]])
                    peaks.append(
                        pd.DataFrame(
                            {
                                "chrom": promoter["chr"],
                                "start": starts,
                                "ends": ends,
                                "gene": gene,
                            }
                        )
                    )
                peaks = pd.concat(peaks)

                peaks_folder = folder_root / "peaks" / dataset_name / peakcaller
                peaks_folder.mkdir(exist_ok=True, parents=True)
                peaks.to_csv(
                    peaks_folder / "peaks.bed", index=False, header=False, sep="\t"
                )
            else:
                peaks_folder = folder_root / "peaks" / dataset_name / peakcaller
                peaks = pd.read_table(
                    peaks_folder / "peaks.bed",
                    names=["chrom", "start", "end"],
                    usecols=[0, 1, 2],
                )

                if peakcaller == "genrich":
                    peaks["start"] += 1

            import pybedtools

            promoters_bed = pybedtools.BedTool.from_dataframe(
                promoters.reset_index()[["chr", "start", "end", "gene"]]
            )
            peaks_bed = pybedtools.BedTool.from_dataframe(peaks)

            # create peaks dataframe
            if peakcaller != "stack":
                intersect = promoters_bed.intersect(peaks_bed)
                intersect = intersect.to_dataframe()

                # peaks = intersect[["score", "strand", "thickStart", "name"]]
                peaks = intersect
            peaks.columns = ["chrom", "start", "end", "gene"]
            peaks = peaks.loc[peaks["start"] != -1]
            peaks.index = pd.Index(
                peaks.chrom
                + ":"
                + peaks.start.astype(str)
                + "-"
                + peaks.end.astype(str),
                name="peak",
            )

            peaks["relative_begin"] = (
                peaks["start"]
                - promoters.loc[peaks["gene"], "start"].values
                + window[0]
            )
            peaks["relative_stop"] = (
                peaks["end"] - promoters.loc[peaks["gene"], "start"].values + window[0]
            )

            peaks["relative_start"] = np.where(
                promoters.loc[peaks["gene"], "strand"] == 1,
                peaks["relative_begin"],
                -peaks["relative_stop"],
            )
            peaks["relative_end"] = np.where(
                promoters.loc[peaks["gene"], "strand"] == -1,
                -peaks["relative_begin"],
                peaks["relative_stop"],
            )

            peaks["gene_ix"] = fragments.var["ix"][peaks["gene"]].values

            peaks["peak"] = peaks.index

            peaks.index = peaks.peak + "_" + peaks.gene
            peaks.index.name = "peak_gene"

            peakcounts.peaks = peaks

            # count
            fragments.obs["ix"] = np.arange(fragments.obs.shape[0])
            peakcounts.count_peaks(
                folder_data_preproc / "atac_fragments.tsv.gz", fragments.obs.index
            )
