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
design["force"] = True


for gene, subdesign in design.groupby("gene", sort=False):
    genes_oi = genes_all == gene
    scores_folder = prediction.path / "scoring" / "pairwindow_gene" / gene
    scores_folder.mkdir(exist_ok=True, parents=True)

    subdesign_row = subdesign.iloc[0]
    force = subdesign_row["force"] or not (scores_folder / "interpolated.pkl").exists()

    if force:
        try:
            scores_folder_window = prediction.path / "scoring" / "window_gene" / gene
            window_scoring = chd.scoring.prediction.Scoring.load(scores_folder_window)

            scores_folder = prediction.path / "scoring" / "pairwindow_gene" / gene
            windowpair_scoring = chd.scoring.prediction.Scoring.load(scores_folder)

            scores_folder_baseline = (
                prediction.path / "scoring" / "pairwindow_gene_baseline" / gene
            )
            windowpair_baseline_scoring = chd.scoring.prediction.Scoring.load(
                scores_folder_baseline
            )
        except FileNotFoundError as e:
            continue
        print(gene)

        retained_additive = pd.Series(
            1
            - (
                (
                    1
                    - (
                        window_scoring.genescores["retained"]
                        .sel(gene=gene)
                        .mean("model")
                        .sel(
                            window=windowpair_scoring.design["window2"].values,
                            phase="validation",
                        )
                    ).values
                )
                + (
                    1
                    - (
                        window_scoring.genescores["retained"]
                        .sel(gene=gene)
                        .mean("model")
                        .sel(
                            window=windowpair_scoring.design["window1"].values,
                            phase="validation",
                        )
                    ).values
                )
            ),
            windowpair_scoring.design.index,
        )

        # because some fragments may be in two windows, we need to use a baseline to correct for this
        additive_baseline = windowpair_baseline_scoring.genescores["cor"].sel(gene=gene)
        additive_base1 = (
            window_scoring.genescores["cor"]
            .sel(gene=gene)
            .sel(window=windowpair_scoring.design["window1"].values)
        ).reindex_like(additive_baseline)
        additive_base2 = (
            window_scoring.genescores["cor"]
            .sel(gene=gene)
            .sel(window=windowpair_scoring.design["window2"].values)
        ).reindex_like(additive_baseline)

        deltacor1 = additive_base1.values - additive_baseline
        deltacor2 = additive_base2.values - additive_baseline
        deltacor_additive = (
            additive_base1.values + additive_base2.values
        ) - 2 * additive_baseline
        deltacor_interacting = (
            windowpair_scoring.genescores["cor"].sel(gene=gene) - additive_baseline
        )
        reldeltacor_interacting = deltacor_interacting - np.minimum(
            deltacor1, deltacor2
        )

        # effect
        # because some fragments may be in two windows, we need to use a baseline to correct for this
        additive_baseline = windowpair_baseline_scoring.genescores["effect"].sel(
            gene=gene
        )
        additive_base1 = (
            window_scoring.genescores["effect"]
            .sel(gene=gene)
            .sel(window=windowpair_scoring.design["window1"].values)
        ).reindex_like(additive_baseline)
        additive_base2 = (
            window_scoring.genescores["effect"]
            .sel(gene=gene)
            .sel(window=windowpair_scoring.design["window2"].values)
        ).reindex_like(additive_baseline)

        effect1 = additive_base1.values - additive_baseline
        effect2 = additive_base2.values - additive_baseline
        effect_additive = (
            additive_base1.values + additive_base2.values
        ) - 2 * additive_baseline / 2
        effect_interacting = windowpair_scoring.genescores["effect"].sel(gene=gene)

        deltaeffect_interacting = effect_interacting - effect_additive

        # create simplified interaction dataframe
        phase = "test"
        # phase = "validation"

        interaction = windowpair_scoring.design.copy()
        interaction["deltacor"] = (
            deltacor_interacting.sel(phase=phase).mean("model").to_pandas()
        )
        interaction["reldeltacor"] = (
            reldeltacor_interacting.sel(phase=phase).mean("model").to_pandas()
        )
        interaction["deltacor1"] = deltacor1.sel(phase=phase).mean("model").to_pandas()
        interaction["deltacor2"] = deltacor2.sel(phase=phase).mean("model").to_pandas()
        interaction["effect"] = (
            effect_interacting.sel(phase=phase).mean("model").to_pandas()
        )

        additive = windowpair_scoring.design.copy()
        additive["deltacor"] = (
            deltacor_additive.sel(phase=phase).mean("model").to_pandas()
        )
        additive["effect"] = effect_additive.sel(phase=phase).mean("model").to_pandas()

        interaction["deltacor_interaction"] = (
            interaction["deltacor"] - additive["deltacor"]
        )
        interaction["effect_interaction"] = interaction["effect"] - additive["effect"]

        # calculate significance
        x = deltacor_interacting.sel(phase="test").values

        import scipy.stats

        def fdr(p_vals):
            from scipy.stats import rankdata

            ranked_p_values = rankdata(p_vals)
            fdr = p_vals * len(p_vals) / ranked_p_values
            fdr[fdr > 1] = 1

            return fdr

        scores_statistical = []
        for i in range(x.shape[1]):
            y = x[:, i]
            if y.std() < 1e-3:
                scores_statistical.append(1)
            else:
                scores_statistical.append(scipy.stats.ttest_1samp(y, 0).pvalue)
        scores_statistical = pd.DataFrame({"pval": scores_statistical})
        scores_statistical["qval"] = fdr(scores_statistical["pval"])
        interaction["qval"] = scores_statistical["qval"].values

        # relative interaction effect
        import scipy.stats

        interaction["deltacor_prod"] = np.prod(
            np.abs(interaction[["deltacor1", "deltacor2"]]), 1
        )

        lm = scipy.stats.linregress(
            interaction["deltacor_prod"], interaction["deltacor_interaction"]
        )

        interaction["deltacor_interaction_corrected"] = (
            interaction["deltacor_interaction"]
            - lm.intercept
            - lm.slope * interaction["deltacor_prod"]
        )

        # save
        interpolated = {
            "interaction": interaction,
        }
        pickle.dump(interpolated, (scores_folder / "interpolated.pkl").open("wb"))
