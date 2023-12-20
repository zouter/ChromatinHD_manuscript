import pandas as pd
import numpy as np
import tqdm.auto as tqdm

import chromatinhd as chd
import chromatinhd.data.peakcounts

import pathlib

from chromatinhd_manuscript.designs_pred import dataset_peakcaller_combinations as design_1
from chromatinhd_manuscript.designs_pred import (
    traindataset_testdataset_peakcaller_combinations as design_2,
)

device = "cuda:1"


design_2["dataset"] = design_2["testdataset"]

design = pd.concat([design_1, design_2])
design.index = np.arange(len(design))

# design = design.loc[~design["peakcaller"].str.startswith("rolling")].copy()
design = design.query("regions in ['10k10k', '100k100k']")
# design = design.query("dataset in ['pbmc10kx', 'pbmc10k_gran', 'pbmc3k']")
design = design.query("dataset == 'liver'")
# design = design.query("peakcaller not in ['1k1k', 'gene_body', 'stack']")
design = design.query("peakcaller in ['rolling_50', 'rolling_100', 'rolling_500']")
# design = design.query("peakcaller == 'rolling_500'")
# design = design.query("peakcaller == 'cellranger'")
# design = design.query("peakcaller == 'genrich'")
# design = design.query("peakcaller == 'macs2_improved'")
# design = design.query("peakcaller == 'macs2_leiden_0.1'")
# design = design.query("peakcaller == 'macs2_leiden_0.1_merged'")
# design = design.query("peakcaller == 'encode_screen'")
# design = design.query("peakcaller == 'cellranger'")
# design = design.query("regions == '10k10k'")
# design = design.query("regions == '100k100k'")
# design = design.query("testdataset == 'pbmc10k_gran-pbmc10k'")
# design = design.query("testdataset == 'lymphoma-pbmc10k'")

##
# design for pbmc10k gene subset (used in ablation)
# design = chd.utils.crossing(
#     pd.DataFrame({"dataset": ["pbmc10k/subsets/top250"]}),
#     pd.DataFrame({"peakcaller": ["macs2_leiden_0.1_merged", "rolling_100", "rolling_500", "rolling_50"]}),
#     pd.DataFrame({"regions": ["10k10k", "20k20k", "100k100k", "200k200k", "500k500k", "1m1m"]}),
# )
# design = chd.utils.crossing(
#     pd.DataFrame({"dataset": ["hspc"]}),
#     # pd.DataFrame({"peakcaller": ["macs2_leiden_0.1_merged", "cellranger", "rolling_500", "rolling_100", "rolling_50", "encode_screen"]}),
#     pd.DataFrame({"peakcaller": ["encode_screen"]}),
#     pd.DataFrame({"regions": ["100k100k"]}),
# )
# design = chd.utils.crossing(
#     pd.DataFrame({"dataset": ["hspc_focus"]}),
#     pd.DataFrame(
#         {
#             "peakcaller": [
#                 "macs2_leiden_0.1_merged",
#                 "cellranger",
#                 "rolling_500",
#                 "rolling_100",
#                 "rolling_50",
#                 "encode_screen",
#             ]
#         }
#     ),
#     pd.DataFrame({"regions": ["500k500k"]}),
# )
# design = pd.DataFrame({"dataset": ["pbmc10k_int"], "peakcaller": ["macs2_leiden_0.1_merged"], "regions": ["10k10k"]})
# design = pd.DataFrame({"dataset": ["hspc_gmp"], "peakcaller": ["encode_screen"], "regions": ["100k100k"]})
# design = pd.DataFrame({"dataset": ["hspc_gmp"], "peakcaller": ["macs2_leiden_0.1_merged"], "regions": ["10k10k"]})
##

htslib_folder = pathlib.Path("/data/peak_free_atac/software/htslib-1.16/")
tabix_location = htslib_folder / "tabix"

design["force"] = False
# design["force"] = True
print(design)

for (dataset_name, regions_name), design_dataset in design.groupby(["dataset", "regions"]):
    folder_data_preproc = chd.get_output() / "data" / dataset_name.split("/")[0]
    dataset_folder = chd.get_output() / "datasets" / dataset_name

    fragments = chd.data.Fragments(dataset_folder / "fragments" / regions_name)
    regions = chd.data.Regions(dataset_folder / "regions" / regions_name)

    for peakcaller, subdesign in design_dataset.groupby("peakcaller"):
        fragments_location = folder_data_preproc / "atac_fragments.tsv.gz"

        print(subdesign.iloc[0])
        if not fragments_location.exists():
            fragments_location = folder_data_preproc / "fragments.tsv.gz"
            if not fragments_location.exists():
                print("No atac fragments found")
                continue

        if peakcaller.startswith("rolling"):
            fragments = chd.data.Fragments(chd.get_output() / "datasets" / dataset_name / "fragments" / regions_name)
            peakcounts = chd.data.peakcounts.Windows.create(
                path=chd.get_output() / "datasets" / dataset_name / "peakcounts" / peakcaller / regions_name,
                fragments=fragments,
                window_size=int(peakcaller.split("_")[1]),
                tabix_location=tabix_location,
                fragments_location=fragments_location,
                reset=True,
            )
        else:
            peakcounts = chd.data.peakcounts.PeakCounts(
                path=chd.get_output() / "datasets" / dataset_name / "peakcounts" / peakcaller / regions_name
            )

            force = subdesign["force"].iloc[0]
            if not peakcounts.counted:
                force = True

            if force:
                # print(subdesign)

                region_coordinates = regions.coordinates.copy()
                window = regions.window

                try:
                    regions.coordinates
                except:
                    print("No regions found")
                    continue

                if peakcaller == "stack":
                    peaks = region_coordinates.reset_index()[["chrom", "start", "end", "gene"]]
                elif peakcaller == "1k1k":
                    peaks = region_coordinates.copy().reset_index()[["chrom", "start", "end", "gene"]]
                    peaks["start"] = peaks["start"] - window[0] - 1000
                    peaks["end"] = peaks["start"] - window[0] + 1000
                elif peakcaller == "gene_body":
                    peaks = pd.DataFrame(
                        [
                            {
                                "chrom": row["chrom"],
                                "start": int(row["start"]) if row["strand"] == -1 else row["tss"],
                                "end": int(row["end"]) if row["strand"] == 1 else row["tss"],
                                "gene": gene,
                                "tss": row["tss"],
                            }
                            for gene, row in region_coordinates.iterrows()
                        ]
                    )
                    peaks["start"] = peaks["tss"].astype(int)
                    peaks["end"] = peaks["end"].astype(int)
                    print(peaks)
                else:
                    peaks_folder = chd.get_output() / "peaks" / dataset_name / peakcaller
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

                regionss = region_coordinates.reset_index()[["chrom", "start", "end", "gene"]]
                regionss["start"] = np.clip(regionss["start"], 0, None)

                # create peaks dataframe
                if (peakcaller != "stack") and (not peakcaller.startswith("rolling")):
                    regionss_bed = pybedtools.BedTool.from_dataframe(regionss)
                    peaks_bed = pybedtools.BedTool.from_dataframe(peaks)

                    intersect = regionss_bed.intersect(peaks_bed)
                    intersect = intersect.to_dataframe()

                    # peaks = intersect[["score", "strand", "thickStart", "name"]]
                    peaks = intersect
                peaks.columns = ["chrom", "start", "end", "gene"]
                peaks = peaks.loc[peaks["start"] != -1]
                peaks.index = pd.Index(
                    peaks.chrom + ":" + peaks.start.astype(str) + "-" + peaks.end.astype(str),
                    name="peak",
                )

                peaks["relative_begin"] = (
                    peaks["start"] - region_coordinates.loc[peaks["gene"], "start"].values + window[0]
                )
                peaks["relative_stop"] = (
                    peaks["end"] - region_coordinates.loc[peaks["gene"], "start"].values + window[0]
                )

                peaks["relative_start"] = np.where(
                    region_coordinates.loc[peaks["gene"], "strand"] == 1,
                    peaks["relative_begin"],
                    -peaks["relative_stop"],
                )
                peaks["relative_end"] = np.where(
                    region_coordinates.loc[peaks["gene"], "strand"] == -1,
                    -peaks["relative_begin"],
                    peaks["relative_stop"],
                )

                peaks["gene_ix"] = fragments.var.index.get_indexer(peaks["gene"])

                peaks["peak"] = peaks.index

                peaks.index = peaks.peak + "_" + peaks.gene
                peaks.index.name = "peak_gene"

                peakcounts.peaks = peaks

                # count
                fragments.obs["ix"] = np.arange(fragments.obs.shape[0])

                peakcounts.count_peaks(
                    fragments_location,
                    fragments.obs.index,
                    tabix_location=tabix_location,
                    do_count=peakcaller not in ["rolling_100", "rolling_50", "rolling_500"],
                )

                print(peakcounts.var is not None)
