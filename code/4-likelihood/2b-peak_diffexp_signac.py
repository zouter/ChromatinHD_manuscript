import pandas as pd
import numpy as np
import torch
import tqdm.auto as tqdm

import chromatinhd as chd
import chromatinhd.data
import chromatinhd.loaders.fragmentmotif
import chromatinhd.loaders.minibatching

import scanpy as sc

import pickle

import itertools
import pathlib

from chromatinhd_manuscript.designs import (
    dataset_latent_peakcaller_diffexp_combinations as design,
)

design = design.query("diffexp == 'signac'")  #!

design = design.query("dataset != 'alzheimer'")
# design = design.query("dataset == 'pbmc10k'")

R_location = "/data/peak_free_atac/software/R-4.2.2/bin/"
signac_script_location = chd.get_code() / "1-preprocessing" / "peaks" / "run_signac.R"

design["force"] = False

test = False
# test = True

for dataset_name, design_dataset in design.groupby("dataset"):
    print(f"{dataset_name=}")
    folder_data_preproc = chd.get_output() / "data" / dataset_name

    promoter_name, window = "10k10k", np.array([-10000, 10000])

    transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")

    for latent_name, subdesign in design_dataset.groupby("latent"):
        latent_folder = folder_data_preproc / "latent"
        latent = pickle.load((latent_folder / (latent_name + ".pkl")).open("rb"))
        cluster_info = pd.read_pickle(latent_folder / (latent_name + "_info.pkl"))

        for peakcaller, subdesign in subdesign.groupby("peakcaller"):
            scores_dir = (
                chd.get_output()
                / "prediction_differential"
                / dataset_name
                / promoter_name
                / latent_name
                / "signac"  ##
                / peakcaller
            )
            scores_dir.mkdir(parents=True, exist_ok=True)

            desired_outputs = [(scores_dir / "slices.pkl")]
            force = subdesign["force"].iloc[0]
            if not all([desired_output.exists() for desired_output in desired_outputs]):
                force = True

            if force:
                try:
                    print(f"{dataset_name=} {latent_name=} {peakcaller=}")
                    peakcounts = chd.peakcounts.FullPeak(
                        folder=chd.get_output()
                        / "peakcounts"
                        / dataset_name
                        / peakcaller
                    )
                    adata_atac = sc.AnnData(
                        peakcounts.counts.astype(np.float32),
                        obs=peakcounts.obs,
                        var=peakcounts.var,
                    )
                    adata_atac.obs["cluster"] = pd.Categorical(
                        latent.columns[np.where(latent.values)[1]],
                        categories=latent.columns,
                    )

                    import tempfile

                    with tempfile.TemporaryDirectory() as tmpdirname:
                        tempfolder = pathlib.Path(tmpdirname)

                        import scipy.io

                        if test:
                            scipy.io.mmwrite(
                                (tempfolder / "counts.mtx"),
                                peakcounts.counts[:, :100],
                            )
                            peakcounts.var.iloc[:100].to_csv(tempfolder / "var.csv")
                            adata_atac.obs.to_csv(tempfolder / "obs.csv")
                        else:
                            scipy.io.mmwrite(
                                (tempfolder / "counts.mtx"),
                                peakcounts.counts,
                            )
                            peakcounts.var.to_csv(tempfolder / "var.csv")
                            adata_atac.obs.to_csv(tempfolder / "obs.csv")

                        # run R script
                        import os

                        os.system(
                            f"{R_location}/Rscript {signac_script_location} {tempfolder}"
                        )

                        import shutil

                        shutil.copy(
                            str(tempfolder / "results.csv"),
                            str(scores_dir / "results.csv"),
                        )

                    peakscores = pd.read_csv(scores_dir / "results.csv", index_col=0)
                    peakscores["cluster"] = pd.Categorical(
                        peakscores["cluster"], categories=cluster_info.index
                    )

                    peakscores = pd.merge(peakscores, peakcounts.peaks, on="peak")
                    peakscores["logfoldchanges"] = peakscores["avg_log2FC"]
                    peakscores["pvals_adj"] = peakscores["p_val_adj"]

                    peakresult = (
                        chromatinhd.differential.DifferentialSlices.from_peakscores(
                            peakscores,
                            window,
                            len(transcriptome.var),
                            logfoldchanges_cutoff=0.1,
                        )
                    )

                    print(scores_dir)
                    pickle.dump(peakresult, (scores_dir / "slices.pkl").open("wb"))
                except BaseException as e:
                    raise e
