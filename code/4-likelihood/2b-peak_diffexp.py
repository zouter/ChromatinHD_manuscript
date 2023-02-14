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

from designs import dataset_latent_peakcaller_combinations as design

# design = design.query("dataset == 'pbmc10k'")
# design = design.query("dataset == 'brain'")
design = design.query("dataset == 'alzheimer'")

design["force"] = False

print(design)

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
                / "scanpy"
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
                    sc.pp.normalize_total(adata_atac)
                    sc.pp.log1p(adata_atac)

                    adata_atac.obs["cluster"] = pd.Categorical(
                        latent.columns[np.where(latent.values)[1]],
                        categories=latent.columns,
                    )
                    sc.tl.rank_genes_groups(adata_atac, "cluster")

                    peakscores = []
                    for cluster_oi in cluster_info.index:
                        peakscores_cluster = (
                            sc.get.rank_genes_groups_df(adata_atac, group=cluster_oi)
                            .rename(columns={"names": "peak", "scores": "score"})
                            .set_index("peak")
                            .assign(cluster=cluster_oi)
                        )
                        peakscores_cluster = peakcounts.peaks.join(
                            peakscores_cluster, on="peak"
                        ).sort_values("score", ascending=False)
                        peakscores.append(peakscores_cluster)
                    peakscores = pd.concat(peakscores)
                    peakscores["cluster"] = pd.Categorical(
                        peakscores["cluster"], categories=cluster_info.index
                    )
                    peakscores["length"] = peakscores["end"] - peakscores["start"]

                    peakresult = (
                        chromatinhd.differential.DifferentialSlices.from_peakscores(
                            peakscores, window, len(transcriptome.var)
                        )
                    )

                    pickle.dump(peakresult, (scores_dir / "slices.pkl").open("wb"))
                except BaseException as e:
                    print(e)
