import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")

import pickle

import scanpy as sc

import torch

import tqdm.auto as tqdm
import xarray as xr

import chromatinhd as chd

device = "cuda:0"
# device = "cpu"

folder_root = chd.get_output()
folder_data = folder_root / "data"

# transcriptome
# dataset_name = "lymphoma"
dataset_name = "pbmc10k"
# dataset_name = "e18brain"
folder_data_preproc = folder_data / dataset_name

transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")

splitter = "permutations_5fold5repeat"
promoter_name, window = "10k10k", np.array([-10000, 10000])
outcome_source = "magic"
prediction_name = "v20"

# fragments
promoters = pd.read_csv(
    folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
)
window_width = window[1] - window[0]

fragments = chd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)
fragments.obs.index.name = "cell"

# create design to run
print(prediction_name)
prediction = chd.flow.Flow(
    chd.get_output()
    / "prediction_positional"
    / dataset_name
    / promoter_name
    / splitter
    / prediction_name
)

genes_all = fragments.var.index
genes_all_oi = fragments.var.index
# genes_all_oi = transcriptome.var.query("symbol == 'TCF3'").index

design = pd.DataFrame({"gene": genes_all_oi})
design["force"] = False


for gene, subdesign in design.groupby("gene", sort=False):
    genes_oi = genes_all == gene
    scores_folder = prediction.path / "scoring" / "multiwindow_gene" / gene
    scores_folder.mkdir(exist_ok=True, parents=True)

    subdesign_row = subdesign.iloc[0]
    force = subdesign_row["force"] or not (scores_folder / "interpolated.pkl").exists()

    if force:
        print(gene)
        try:
            multiwindow_scoring = chd.scoring.prediction.Scoring.load(scores_folder)
        except FileNotFoundError as e:
            continue

        import scipy.stats

        def fdr(p_vals):
            from scipy.stats import rankdata

            ranked_p_values = rankdata(p_vals)
            fdr = p_vals * len(p_vals) / ranked_p_values
            fdr[fdr > 1] = 1

            return fdr

        x = (
            multiwindow_scoring.genescores["deltacor"]
            .sel(gene=gene, phase=["test", "validation"])
            .stack({"model_phase": ["model", "phase"]})
            .values.T
        )
        scores_statistical = []
        for i in range(x.shape[1]):
            scores_statistical.append(
                scipy.stats.ttest_1samp(x[:, i], 0, alternative="less").pvalue
            )
        scores_statistical = pd.DataFrame({"pvalue": scores_statistical})
        scores_statistical["qval"] = fdr(scores_statistical["pvalue"])

        plotdata = (
            multiwindow_scoring.genescores.mean("model")
            .sel(gene=gene)
            .stack()
            .to_dataframe()
        )
        plotdata = multiwindow_scoring.design.join(plotdata)

        plotdata.loc["validation", "qval"] = scores_statistical["qval"].values
        plotdata.loc["test", "qval"] = scores_statistical["qval"].values

        window_sizes_info = pd.DataFrame(
            {"window_size": multiwindow_scoring.design["window_size"].unique()}
        ).set_index("window_size")
        window_sizes_info["ix"] = np.arange(len(window_sizes_info))

        # interpolate
        positions_oi = np.arange(*window)

        deltacor_test_interpolated = np.zeros(
            (len(window_sizes_info), len(positions_oi))
        )
        deltacor_validation_interpolated = np.zeros(
            (len(window_sizes_info), len(positions_oi))
        )
        retained_interpolated = np.zeros((len(window_sizes_info), len(positions_oi)))
        lost_interpolated = np.zeros((len(window_sizes_info), len(positions_oi)))
        for window_size, window_size_info in window_sizes_info.iterrows():
            # for window_size, window_size_info in window_sizes_info.query(
            #     "window_size == 200"
            # ).iterrows():
            plotdata_oi = plotdata.query("phase in ['validation']").query(
                "window_size == @window_size"
            )

            x = plotdata_oi["window_mid"].values.copy()
            y = plotdata_oi["deltacor"].values.copy()
            y[plotdata_oi["qval"] > 0.1] = 0.0
            deltacor_interpolated_ = np.clip(
                np.interp(positions_oi, x, y) / window_size * 1000,
                -np.inf,
                0,
                # np.inf,
            )
            deltacor_validation_interpolated[
                window_size_info["ix"], :
            ] = deltacor_interpolated_
            plotdata_oi = plotdata.query("phase in ['test']").query(
                "window_size == @window_size"
            )
            x = plotdata_oi["window_mid"].values.copy()
            y = plotdata_oi["deltacor"].values.copy()
            y[plotdata_oi["qval"] > 0.1] = 0.0
            deltacor_interpolated_ = np.clip(
                np.interp(positions_oi, x, y) / window_size * 1000,
                -np.inf,
                0,
                # np.inf,
            )
            deltacor_test_interpolated[
                window_size_info["ix"], :
            ] = deltacor_interpolated_

            retained_interpolated_ = (
                np.interp(
                    positions_oi, plotdata_oi["window_mid"], plotdata_oi["retained"]
                )
                / window_size
                * 1000
            )
            retained_interpolated[window_size_info["ix"], :] = retained_interpolated_
            lost_interpolated_ = (
                np.interp(positions_oi, plotdata_oi["window_mid"], plotdata_oi["lost"])
                / window_size
                * 1000
            )
            lost_interpolated[window_size_info["ix"], :] = lost_interpolated_

        # save
        interpolated = {
            "deltacor_validation": deltacor_validation_interpolated,
            "deltacor_test": deltacor_test_interpolated,
            "retained": retained_interpolated,
            "lost": lost_interpolated,
        }
        pickle.dump(interpolated, (scores_folder / "interpolated.pkl").open("wb"))
