import pandas as pd
import numpy as np
import torch
import tqdm.auto as tqdm

import chromatinhd as chd
import chromatinhd.data

import pickle
import copy

device = "cuda:0"

folder_root = chd.get_output()
folder_data = folder_root / "data"


class Prediction(chd.flow.Flow):
    pass


print(torch.cuda.memory_allocated(0))

from chromatinhd_manuscript.designs import (
    dataset_splitter_method_combinations as design,
)

# design = design.loc[~design["dataset"].isin(["alzheimer"])]
design = design.query("splitter == 'random_5fold'")
# design = design.loc[
#     design["splitter"].isin(["leiden_0.1", "celltype", "overexpression"])
# ]
design = design.query("method == 'v20_initdefault'")
design = design.query("dataset == 'pbmc10k'")

design = design.copy()
design["force"] = False

for dataset_name, subdesign in design.groupby("dataset"):
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

    transcriptome = chromatinhd.data.Transcriptome(
        folder_data_preproc / "transcriptome"
    )

    print(fragments.n_genes)

    for splitter, subdesign in subdesign.groupby("splitter"):
        # create design to run
        from design import get_design, get_folds_training

        methods_info = get_design(transcriptome, fragments)

        # fold_slice = slice(0, 1)
        # fold_slice = slice(0, 5)
        fold_slice = slice(None, None)

        # folds & minibatching
        folds = pickle.load((fragments.path / "folds" / (splitter + ".pkl")).open("rb"))

        for method_name, subdesign in subdesign.groupby("method"):
            method_info = methods_info[method_name]

            prediction = Prediction(
                chd.get_output()
                / "prediction_positional"
                / dataset_name
                / promoter_name
                / splitter
                / method_name
            )

            print(subdesign)

            models = []
            for fold_ix, fold in [
                (fold_ix, fold) for fold_ix, fold in enumerate(folds)
            ][fold_slice]:
                # check if outputs are already there
                desired_outputs = [prediction.path / ("model_" + str(fold_ix) + ".pkl")]
                force = subdesign["force"].iloc[0]
                if not all(
                    [desired_output.exists() for desired_output in desired_outputs]
                ):
                    force = True

                if force:
                    fold = get_folds_training(fragments, [copy.copy(fold)])[0]

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
                    loaders_validation = chd.loaders.LoaderPool(
                        method_info["loader_cls"],
                        method_info["loader_parameters"],
                        n_workers=5,
                    )
                    loaders_validation.shuffle_on_iter = False

                    # model
                    model = method_info["model_cls"](**method_info["model_parameters"])

                    # optimization
                    optimize_every_step = 1
                    lr = 1e-2
                    optimizer = chd.optim.SparseDenseAdam(
                        model.parameters_sparse(),
                        model.parameters_dense(),
                        lr=lr,
                        weight_decay=1e-5,
                    )
                    n_epochs = (
                        30 if "n_epoch" not in method_info else method_info["n_epoch"]
                    )
                    checkpoint_every_epoch = 1

                    # train
                    from chromatinhd.models.positional.trainer import Trainer

                    def paircor(x, y, dim=0, eps=0.1):
                        divisor = (y.std(dim) * x.std(dim)) + eps
                        cor = (
                            (x - x.mean(dim, keepdims=True))
                            * (y - y.mean(dim, keepdims=True))
                        ).mean(dim) / divisor
                        return cor

                    loss = lambda x, y: -paircor(x, y).mean() * 100

                    outcome = transcriptome.X.dense()

                    trainer = Trainer(
                        model,
                        loaders,
                        loaders_validation,
                        optim=optimizer,
                        outcome=outcome,
                        loss=loss,
                        checkpoint_every_epoch=checkpoint_every_epoch,
                        optimize_every_step=optimize_every_step,
                        n_epochs=n_epochs,
                        device=device,
                    )
                    trainer.train(
                        fold["minibatches_train_sets"],
                        fold["minibatches_validation_trace"],
                    )

                    model = model.to("cpu")
                    pickle.dump(
                        model,
                        open(
                            prediction.path / ("model_" + str(fold_ix) + ".pkl"), "wb"
                        ),
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
                    fig.savefig(
                        str(prediction.path / ("trace_" + str(fold_ix) + ".png"))
                    )
                    plt.close()
