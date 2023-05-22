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

from chromatinhd_manuscript.designs import dataset_peakcaller_combinations as design_1
from chromatinhd_manuscript.designs import (
    traindataset_testdataset_peakcaller_combinations as design_2,
)

design_2["dataset"] = design_2["testdataset"]

design = pd.concat([design_1, design_2])
design.index = np.arange(len(design))

# design = design.loc[~design["peakcaller"].str.startswith("rolling")].copy()
# design = design.query("dataset == 'morf_20'").copy()
# design = design.query("peakcaller == '1k1k'").copy()
# design = design.query("peakcaller == 'stack'").copy()
# design = design.loc[
#     ~((design["dataset"] == "alzheimer") & (design["peakcaller"] == "genrich"))
# ]
design = design.query("dataset == 'pbmc10k'")
design = design.query("promoter == '100k100k'")

design["force"] = False
print(design)

for (dataset_name, promoter), design_dataset in design.groupby(["dataset", "promoter"]):
    # transcriptome
    folder_data_preproc = folder_data / dataset_name

    # fragments
    # promoter_name, window = "1k1k", np.array([-1000, 1000])
    if promoter == "10k10k":
        window = np.array([-10000, 10000])
    elif promoter == "100k100k":
        window = np.array([-100000, 100000])

    fragments = chd.data.Fragments(folder_data_preproc / "fragments" / promoter)
    fragments.window = window

    for peakcaller, subdesign in design_dataset.groupby("peakcaller"):
        peakcounts = chd.peakcounts.FullPeak(
            folder=chd.get_output() / "peakcounts" / dataset_name / peakcaller
        )

        desired_outputs = [peakcounts.path / ("counts.pkl")]
        force = subdesign["force"].iloc[0]
        if not all([desired_output.exists() for desired_output in desired_outputs]):
            force = True

        if force:
            print(subdesign)
            promoters = pd.read_csv(
                folder_data_preproc / ("promoters_" + promoter + ".csv"),
                index_col=0,
            )

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
            elif peakcaller == "1k1k":
                peaks = promoters.copy().reset_index()[["chr", "start", "end", "gene"]]
                peaks["start"] = peaks["start"] - window[0] - 1000
                peaks["end"] = peaks["start"] - window[0] + 1000
            else:
                peaks_folder = folder_root / "peaks" / dataset_name / peakcaller
                try:
                    peaks = pd.read_table(
                        peaks_folder / "peaks.bed",
                        names=["chrom", "start", "end"],
                        usecols=[0, 1, 2],
                    )
                except FileNotFoundError as e:
                    print(e)
                    continue

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

            fragments_location = folder_data_preproc / "atac_fragments.tsv.gz"
            if not fragments_location.exists():
                fragments_location = folder_data_preproc / "fragments.tsv.gz"
                if not fragments_location.exists():
                    print("No atac fragments found")
                    continue
            peakcounts.count_peaks(fragments_location, fragments.obs.index)
