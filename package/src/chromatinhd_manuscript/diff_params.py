import chromatinhd as chd
import hashlib
import copy

params = []

params.append(
    dict(
        model_params=dict(
            encoder="shared",
            encoder_params=dict(
                binwidths=(5000, 1000, 500, 100, 50, 25),
            ),
        ),
        train_params=dict(n_cells_step=5000, early_stopping=False, n_epochs=150),
        label="v30",
    ),
)


params.append(
    dict(
        model_params=dict(
            encoder="shared",
            encoder_params=dict(
                binwidths=(5000, 1000, 500, 100, 50, 25),
            ),
        ),
        train_params=dict(n_cells_step=5000, early_stopping=False, n_epochs=150, lr=1e-3),
        label="v31",
    ),
)

labels = []
for param in params:
    param["hex"] = hashlib.md5(str({k: v for k, v in param.items() if k not in ["label"]}).encode()).hexdigest()
    if "label" in param:
        labels.append(param["label"])
    else:
        labels.append(param["hex"])

params = {h: v for h, v in zip(labels, params)}
