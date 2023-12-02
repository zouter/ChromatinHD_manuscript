import pandas as pd
import numpy as np
import torch

import chromatinhd as chd
import chromatinhd.data
import matplotlib.pyplot as plt

import pickle

device = "cuda:0"

folder_root = chd.get_output()
folder_data = folder_root / "data"


class Prediction(chd.flow.Flow):
    pass


print(torch.cuda.memory_allocated(0))

from chromatinhd_manuscript.designs_pred import (
    dataset_folds_method_combinations as design,
)

design = design.query("folds == '5x5'")
# design = design.query("splitter == 'permutations_5fold5repeat'")
# design = design.loc[design["method"].isin(["v20", "counter"])]
design = design.query("method == 'v22'")
design = design.query("dataset == 'pbmc10k'")
# design = design.query("dataset == 'pbmc3k'")
design = design.query("regions == '100k100k'")
design = design.query("layer == 'normalized'")
# design = design.query("layer == 'magic'")
# design = design.query("layer == 'magic'")
# design = design.query("regions == '20kpromoter'")
# design = design.query("regions == '100k100k'")

design = design.copy()
design["force"] = True

for (dataset_name, regions_name), subdesign in design.groupby(["dataset", "regions"]):
    print(f"{dataset_name=}")
    dataset_folder = chd.get_output() / "datasets" / dataset_name

    fragments = chromatinhd.data.Fragments(dataset_folder / "fragments" / regions_name)
    transcriptome = chromatinhd.data.Transcriptome(dataset_folder / "transcriptome")

    for (folds_name, layer), subdesign in subdesign.groupby(["folds", "layer"]):
        # create design to run
        from chromatinhd.models.pred.model.design import get_design

        methods_info = get_design(transcriptome, fragments)

        # folds & minibatching
        folds = chd.data.folds.Folds(dataset_folder / "folds" / folds_name)

        for method_name, subdesign in subdesign.groupby("method"):
            method_info = methods_info[method_name]

            if "expression_source" in method_info:
                outcome_source = method_info["expression_source"]

            prediction = chd.flow.Flow(
                chd.get_output() / "pred" / dataset_name / regions_name / folds_name / layer / method_name
            )

            prediction_base = chd.flow.Flow(
                chd.get_output() / "pred" / dataset_name / regions_name / folds_name / layer / "v20"
            )

            print(subdesign)

            models = []
            for fold_ix, fold in [(fold_ix, fold) for fold_ix, fold in enumerate(folds)]:
                # check if outputs are already there
                desired_outputs = [prediction.path / str(fold_ix)]
                force = subdesign["force"].iloc[0]
                if not all([desired_output.exists() for desired_output in desired_outputs]):
                    force = True

                base_model = chd.flow.Flow.from_path(prediction_base.path / str(fold_ix))

                if force:
                    # model
                    model = method_info["model_cls"](
                        path=prediction.path / str(fold_ix),
                        base_model=base_model,
                        **method_info["model_parameters"],
                    )

                    model.train_model(n_epochs=30)

                    model.save_state()

                    import gc

                    gc.collect()
                    torch.cuda.empty_cache()

                    fig, ax = plt.subplots()
                    plotdata_validation = (
                        pd.DataFrame(model.trace.validation_steps).groupby("checkpoint").mean().reset_index()
                    )
                    plotdata_train = pd.DataFrame(model.trace.train_steps).groupby("checkpoint").mean().reset_index()
                    ax.plot(
                        plotdata_validation["checkpoint"],
                        plotdata_validation["loss"],
                        label="validation",
                    )
                    # ax.plot(plotdata_train["checkpoint"], plotdata_train["loss"], label = "train")
                    ax.legend()
                    fig.savefig(str(prediction.path / ("trace_" + str(fold_ix) + ".png")))
                    plt.close()
