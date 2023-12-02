import pandas as pd
import numpy as np
import torch
import tqdm.auto as tqdm

import chromatinhd as chd
import chromatinhd.data
import chromatinhd.loaders.fragmentmotif
import chromatinhd.loaders.minibatches

import scanpy as sc

import pickle

import itertools

from chromatinhd_manuscript.designs import (
    dataset_latent_peakcaller_diffexp_combinations as design,
)

# design = design.query("diffexp == 'scanpy_logreg'")  #!
# design = design.query("diffexp == 'scanpy_wilcoxon'")  #!
design = design.query("diffexp == 'scanpy'")  #!

# design = design.query("dataset != 'alzheimer'")
# design = design.query("dataset == 'morf_20'")
# design = design.query("peakcaller == 'encode_screen'")
# design = design.query("dataset == 'alzheimer'")
design = design.query("dataset == 'lymphoma'")
# design = design.query("peakcaller == 'cellranger'")
design = design.query("promoter == '10k10k'")
# design = design.query("dataset == 'pbmc10k_eqtl'")

design["force"] = True

print(design)

for dataset_name, design_dataset in design.groupby("dataset"):
    print(f"{dataset_name=}")
    folder_data_preproc = chd.get_output() / "data" / dataset_name

    promoter_name, window = "10k10k", np.array([-10000, 10000])

    promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0)

    for latent_name, subdesign in design_dataset.groupby("latent"):
        latent_folder = folder_data_preproc / "latent"
        latent = pickle.load((latent_folder / (latent_name + ".pkl")).open("rb"))
        cluster_info = pd.read_pickle(latent_folder / (latent_name + "_info.pkl"))

        for (peakcaller, diffexp), subdesign in subdesign.groupby(["peakcaller", "diffexp"]):
            scores_dir = (
                chd.get_output()
                / "prediction_differential"
                / dataset_name
                / promoter_name
                / latent_name
                / diffexp
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
                        folder=chd.get_output() / "peakcounts" / dataset_name / peakcaller
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

                    if diffexp == "scanpy_logreg":
                        sc.pp.scale(adata_atac)
                    sc.tl.rank_genes_groups(
                        adata_atac,
                        "cluster",
                        method={
                            "scanpy": "t-test",
                            "scanpy_logreg": "logreg",
                            "scanpy_wilcoxon": "wilcoxon",
                        }[diffexp],
                        max_iter=500,
                    )

                    peakscores = []
                    for cluster_oi in cluster_info.index:
                        if diffexp == "scanpy_logreg":
                            peakscores_cluster = (
                                pd.DataFrame(
                                    {
                                        "logfoldchanges": adata_atac.uns["rank_genes_groups"]["scores"][cluster_oi]
                                        * 10,
                                        "peak": adata_atac.uns["rank_genes_groups"]["names"][cluster_oi],
                                    }
                                )
                                .set_index("peak")
                                .assign(cluster=cluster_oi)
                            )
                            peakscores_cluster["score"] = peakscores_cluster["logfoldchanges"]
                            peakscores_cluster["pvals_adj"] = 0.0
                            # import IPython

                            # IPython.embed()
                            # raise ValueError()
                        elif diffexp in ["scanpy", "scanpy_wilcoxon"]:
                            peakscores_cluster = (
                                sc.get.rank_genes_groups_df(adata_atac, group=cluster_oi)
                                .rename(columns={"names": "peak", "scores": "score"})
                                .set_index("peak")
                                .assign(cluster=cluster_oi)
                            )
                        else:
                            raise ValueError(diffexp)
                        peakscores_cluster = peakcounts.peaks.join(peakscores_cluster, on="peak").sort_values(
                            "score", ascending=False
                        )
                        peakscores.append(peakscores_cluster)
                    peakscores = pd.concat(peakscores)
                    peakscores["cluster"] = pd.Categorical(peakscores["cluster"], categories=cluster_info.index)
                    peakscores["length"] = peakscores["end"] - peakscores["start"]

                    peakresult = chromatinhd.differential.DifferentialSlices.from_peakscores(
                        peakscores, window, len(promoters.index)
                    )

                    pickle.dump(peakresult, (scores_dir / "slices.pkl").open("wb"))

                except KeyboardInterrupt as e:
                    raise e
                except BaseException as e:
                    print("ERROR: ", e)
                    raise e
