import pandas as pd
import numpy as np
import torch
import tqdm.auto as tqdm

import chromatinhd as chd
import chromatinhd.data
import chromatinhd.loaders.fragmentmotif
import chromatinhd.loaders.minibatching

import pickle

device = "cuda:0"

folder_root = chd.get_output()
folder_data = folder_root / "data"


class Prediction(chd.flow.Flow):
    pass


# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:4096'

print(torch.cuda.memory_allocated(0))

import itertools

design = pd.DataFrame.from_records(
    itertools.chain(
        # itertools.product(
        #     ["lymphoma"],
        #     ["celltype"],
        #     ["v4_128-64-32_30_rep"]
        # ),
        itertools.product(
            [
                "pbmc10k",
                # "e18brain",
                # "brain",
                # "alzheimer"
            ],
            ["leiden_0.1"],
            # ["v4_128-64-32_30_rep"],
            ["v5_128-64-32"],
        ),
    ),
    columns=["dataset", "latent", "method"],
)
print(design)

for dataset_name, design_dataset in design.groupby("dataset"):
    print(f"{dataset_name=}")
    folder_data_preproc = folder_data / dataset_name

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

    print(fragments.n_genes)

    for latent_name, design_latent in design_dataset.groupby("latent"):
        # create design to run
        from design import get_design, get_folds_training

        methods_info = get_design(dataset_name, latent_name, fragments)

        fold_slice = slice(0, 1)
        # fold_slice = slice(0, 5)

        # folds & minibatching
        folds = pickle.load((fragments.path / "folds.pkl").open("rb"))
        folds = get_folds_training(fragments, folds)

        for method_name, method_row in design_latent.groupby("method"):
            method_info = methods_info[method_name]

            print(f"{dataset_name=} {promoter_name=} {method_name=}")
            prediction = Prediction(
                chd.get_output()
                / "prediction_likelihood"
                / dataset_name
                / promoter_name
                / latent_name
                / method_name
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
                method_info["loader_cls"],
                method_info["loader_parameters"],
                shuffle_on_iter=True,
                n_workers=10,
            )
            print("haha!")
            loaders_validation = chd.loaders.LoaderPool(
                method_info["loader_cls"], method_info["loader_parameters"], n_workers=5
            )
            loaders_validation.shuffle_on_iter = False

            models = []
            for fold_ix, fold in [
                (fold_ix, fold) for fold_ix, fold in enumerate(folds)
            ][fold_slice]:
                # model
                print(torch.cuda.memory_allocated(0))
                model = method_info["model_cls"](
                    **method_info["model_parameters"],
                    # loader = loaders.loaders[0]
                )
                print(">>", torch.cuda.memory_allocated(0))
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
                n_epochs = (
                    10 if "n_epoch" not in method_info else method_info["n_epoch"]
                )
                checkpoint_every_epoch = 1

                # initialize loaders
                loaders.initialize(next_task_sets=fold["minibatches_train_sets"])
                loaders_validation.initialize(fold["minibatches_validation_trace"])

                # train
                import chromatinhd.train

                trainer = chd.train.Trainer(
                    model,
                    loaders,
                    loaders_validation,
                    optimizer,
                    checkpoint_every_epoch=checkpoint_every_epoch,
                    optimize_every_step=optimize_every_step,
                    n_epochs=n_epochs,
                    device=device,
                )
                trainer.train()

                model = model.to("cpu")
                pickle.dump(
                    model,
                    open(prediction.path / ("model_" + str(fold_ix) + ".pkl"), "wb"),
                )

                torch.cuda.empty_cache()
                import gc

                gc.collect()
                torch.cuda.empty_cache()

                import matplotlib.pyplot as plt

                fig, ax = plt.subplots()
                plotdata_validation = (
                    pd.DataFrame(trainer.trace.validation_steps)
                    .groupby("checkpoint")
                    .mean()
                    .reset_index()
                )
                plotdata_train = (
                    pd.DataFrame(trainer.trace.train_steps)
                    .groupby("checkpoint")
                    .mean()
                    .reset_index()
                )
                ax.plot(
                    plotdata_validation["checkpoint"],
                    plotdata_validation["loss"],
                    label="validation",
                )
                # ax.plot(plotdata_train["checkpoint"], plotdata_train["loss"], label = "train")
                ax.legend()
                fig.savefig(str(prediction.path / ("trace_" + str(fold_ix) + ".png")))
                plt.close()
