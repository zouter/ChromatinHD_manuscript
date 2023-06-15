import pandas as pd
import numpy as np
import torch
import tqdm.auto as tqdm

import chromatinhd as chd
import chromatinhd.data
import chromatinhd.loaders.fragmentmotif
import chromatinhd.loaders.minibatching

import pickle

import matplotlib.pyplot as plt

device = "cuda:0"

folder_root = chd.get_output()
folder_data = folder_root / "data"

import torch_scatter


class GeneLikelihoodHook:
    def __init__(self, n_genes):
        self.n_genes = n_genes
        self.likelihood_mixture = []
        self.likelihood_counts = []

    def start(self):
        self.likelihood_mixture_checkpoint = np.zeros(self.n_genes)
        self.likelihood_counts_checkpoint = np.zeros(self.n_genes)
        return {}

    def run_individual(self, model, data):
        self.likelihood_mixture_checkpoint[data.genes_oi] += (
            torch_scatter.scatter_sum(
                model.track["likelihood_mixture"],
                data.cut_local_gene_ix,
                dim_size=data.n_genes,
            )
            .detach()
            .cpu()
            .numpy()
        )
        self.likelihood_counts_checkpoint[data.genes_oi] += (
            model.track["likelihood_fragmentcounts"].sum(0).detach().cpu().numpy()
        )

    def finish(self):
        self.likelihood_mixture.append(self.likelihood_mixture_checkpoint)
        self.likelihood_counts.append(self.likelihood_counts_checkpoint)


class EmbeddingHook:
    def __init__(self, n_cells, n_latent_dimensions, loaders_all):
        self.n_cells = n_cells
        self.n_latent_dimensions = n_latent_dimensions
        self.loaders_all = loaders_all
        self.embeddings_checkpoint = []

    def run(self, model):
        embedding = np.zeros((self.n_cells, self.n_latent_dimensions))

        self.loaders_all.restart()
        for data in self.loaders_all:
            data = data.to(device)
            with torch.no_grad():
                embedding[data.cells_oi] = (
                    model.evaluate_latent(data).detach().cpu().numpy()
                )

            self.loaders_all.submit_next()
        self.embeddings_checkpoint.append(embedding)


class Prediction(chd.flow.Flow):
    pass


for dataset_name in [
    "e18brain",
    "lymphoma",
    "pbmc10k",
]:
    print(f"{dataset_name=}")
    # transcriptome
    folder_data_preproc = folder_data / dataset_name

    transcriptome = chromatinhd.data.Transcriptome(
        folder_data_preproc / "transcriptome"
    )

    # fragments
    # promoter_name, window = "1k1k", np.array([-1000, 1000])
    promoter_name, window = "10k10k", np.array([-10000, 10000])
    # promoter_name, window = "20kpromoter", np.array([-10000, 0])
    promoters = pd.read_csv(
        folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
    )
    window_width = window[1] - window[0]

    fragments = chromatinhd.data.Fragments(
        folder_data_preproc / "fragments" / promoter_name
    )
    fragments.window = window
    fragments.create_cut_data()

    # create design to run
    from design import get_design, get_folds_training

    design = get_design(fragments)
    design = {
        k: design[k]
        for k in [
            # "v4",
            # "v4_baseline",
            # "v4_decoder1",
            # "v4_1freq",
            # "v5",
            # "v5_baseline",
            # "v5_1decoder",
            # "v5_8",
            # "v5_32",
            # "v5_norescale",
            # "v5_encoder32",
            # "v5_regularizefragmentcounts",
            # "v5_regularizefragmentcounts_400epoch",
            # "v5_s0.8",
            # "v5_s0.5",
            # "v5_s0.3",
            # "v5_mixtureautoscale",
            # "v5_mixturescale0.1",
            # "v5_mixturelaplace",
            # "v5_mixturelaplace",
            "v5_2encoder",
            # "v6",
        ]
    }
    fold_slice = slice(0, 1)

    # folds & minibatching
    folds = pickle.load((fragments.path / "folds.pkl").open("rb"))
    folds = get_folds_training(fragments, folds)

    for prediction_name, design_row in design.items():
        print(f"{dataset_name=} {promoter_name=} {prediction_name=}")
        prediction = chd.flow.Flow(
            chd.get_output()
            / "prediction_vae"
            / dataset_name
            / promoter_name
            / prediction_name
        )

        # loaders
        print("collecting...")
        if "loaders" in globals():
            globals()["loaders"].terminate()
            del globals()["loaders"]
            import gc

            gc.collect()
        if "loaders_validation" in globals():
            globals()["loaders_validation"].terminate()
            del globals()["loaders_validation"]
            import gc

            gc.collect()
        print("collected")
        loaders = chd.loaders.LoaderPool(
            design_row["loader_cls"],
            design_row["loader_parameters"],
            n_workers=10,
            shuffle_on_iter=True,
        )
        print("haha!")
        loaders_validation = chd.loaders.LoaderPool(
            design_row["loader_cls"],
            design_row["loader_parameters"],
            n_workers=5,
            shuffle_on_iter=False,
        )
        loaders_all = chd.loaders.LoaderPool(
            design_row["loader_cls"],
            design_row["loader_parameters"],
            n_workers=5,
            shuffle_on_iter=False,
        )

        models = []
        for fold_ix, fold in [(fold_ix, fold) for fold_ix, fold in enumerate(folds)][
            fold_slice
        ]:
            # model
            model = design_row["model_cls"](**design_row["model_parameters"])
            # model = pickle.load(open(prediction.path / ("model_" + str(fold_ix) + ".pkl"), "rb"))

            # optimization
            optimize_every_step = 1
            lr = 1e-3
            optimizer = chd.optim.SparseDenseAdam(
                model.parameters_sparse(),
                model.parameters_dense(),
                lr=lr,
                weight_decay=1e-5,
            )
            n_epochs = 200
            if "n_epochs" in design_row:
                n_epochs = design_row["n_epochs"]
            checkpoint_every_epoch = 3

            # initialize loaders
            loaders.initialize(next_task_sets=fold["minibatches_train_sets"])
            loaders_validation.initialize(fold["minibatches_validation_trace"])
            loaders_all.initialize(fold["minibatches_all"])

            # initialize hooks
            hook_genelikelihood = GeneLikelihoodHook(fragments.n_genes)
            hook_embedding = EmbeddingHook(
                fragments.n_cells, model.n_latent_dimensions, loaders_all
            )
            hooks_checkpoint = [hook_genelikelihood]
            hooks_checkpoint2 = [hook_embedding]

            # train
            import chromatinhd.train

            outcome = transcriptome.X.dense()
            trainer = chd.train.Trainer(
                model,
                loaders,
                loaders_validation,
                optimizer,
                checkpoint_every_epoch=checkpoint_every_epoch,
                optimize_every_step=optimize_every_step,
                n_epochs=n_epochs,
                device=device,
                hooks_checkpoint=hooks_checkpoint,
                hooks_checkpoint2=hooks_checkpoint2,
            )
            trainer.train()

            model = model.to("cpu")
            pickle.dump(
                model, open(prediction.path / ("model_" + str(fold_ix) + ".pkl"), "wb")
            )

            ##
            likelihood_mixture = pd.DataFrame(
                np.vstack(hook_genelikelihood.likelihood_mixture),
                columns=fragments.var.index,
            ).T
            likelihood_counts = pd.DataFrame(
                np.vstack(hook_genelikelihood.likelihood_counts),
                columns=fragments.var.index,
            ).T

            fig, axes = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={"wspace": 0.5})
            for ax, plotdata_mixture, plotdata_counts in [
                [axes[0], likelihood_mixture.mean(), likelihood_counts.mean()],
                [axes[1], likelihood_mixture.mean()[4:], likelihood_counts.mean()[4:]],
                [
                    axes[2],
                    likelihood_mixture.mean()[-50:],
                    likelihood_counts.mean()[-50:],
                ],
            ]:
                plotdata_mixture.plot(color="green", label="mixture", ax=ax)
                plt.legend()
                ax2 = ax.twinx()
                plotdata_counts.plot(color="red", label="counts", ax=ax2)

            fig.savefig(str(prediction.path / ("trace_" + str(fold_ix) + ".png")))
            plt.close()

            ##
            pickle.dump(
                hook_embedding.embeddings_checkpoint,
                (prediction.path / (f"embeddings_checkpoint_{fold_ix}.pkl")).open("wb"),
            )

            torch.cuda.empty_cache()
            import gc

            gc.collect()
            torch.cuda.empty_cache()
