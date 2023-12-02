import chromatinhd as chd
import chromatinhd.models.pred.model.better
import hashlib
import copy

params = []

params.append(
    dict(
        model_params=dict(
            encoder="shared",
            encoder_params=dict(
                binwidths=(5000, 1000, 500, 200, 100, 50),
            ),
        ),
        train_params=dict(),
        label="binary_1000-50bw",
    ),
)

params.append(
    dict(
        model_params=dict(
            encoder="shared",
            encoder_params=dict(
                binwidths=(5000, 1000, 500, 200, 100, 50),
                delta_regularization=False,
            ),
        ),
        train_params=dict(),
        label="binary_1000-50bw_wdeltareg-no",
    ),
)


params.append(
    dict(
        model_params=dict(
            encoder="shared",
            encoder_params=dict(
                binwidths=(5000, 1000, 500, 200, 100, 50),
                differential=False,
            ),
        ),
        train_params=dict(),
        label="binary_1000-50bw_nodiff",
    ),
)

params.append(
    dict(
        model_params=dict(
            encoder="shared",
            encoder_params=dict(
                binwidths=(50,),
            ),
        ),
        train_params=dict(),
        label="binary_50bw",
    ),
)


params.append(
    dict(
        model_params=dict(
            encoder="shared",
            encoder_params=dict(
                binwidths=(5000, 1000, 500, 200, 100, 50),
            ),
        ),
        train_params=dict(early_stopping=False),
        label="binary_1000-50bw_earlystop-no",
    ),
)

params.append(
    dict(
        model_params=dict(
            encoder="shared",
            encoder_params=dict(
                binwidths=(5000, 1000, 500, 200, 100, 50, 25),
            ),
        ),
        train_params=dict(),
        label="binary_1000-25bw",
    ),
)

params.append(
    dict(
        model_params=dict(
            encoder="shared",
            encoder_params=dict(
                binwidths=(25,),
            ),
        ),
        train_params=dict(),
        label="binary_25bw",
    ),
)

binwidths = (5000, 1000, 500, 200, 100, 50, 25)
binwidth_titration_ids = []
for i in range(1, len(binwidths)):
    params.append(
        dict(
            model_params=dict(
                encoder="shared",
                encoder_params=dict(
                    binwidths=binwidths[i:],
                ),
            ),
            train_params=dict(),
            label=f"binary_{binwidths[i:]}bw",
        ),
    )
    binwidth_titration_ids.append(f"binary_{binwidths[i:]}bw")


w_delta_p_scales = (0.1, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 20.0)
w_delta_p_scale_titration_ids = []
for w_delta_p_scale in w_delta_p_scales:
    params.append(
        dict(
            model_params=dict(
                encoder="shared",
                encoder_params=dict(
                    delta_p_scale=w_delta_p_scale,
                ),
            ),
            train_params=dict(),
            label=f"binary_wdeltareg-{w_delta_p_scale}",
        ),
    )
    w_delta_p_scale_titration_ids.append(params[-1]["label"])


params.append(
    dict(
        model_params=dict(
            encoder="split",
            encoder_params=dict(
                binwidths=(5000, 1000, 500, 100, 50, 25),
            ),
        ),
        train_params=dict(),
        label="binary_split_[5k,1k,500,100,50,25]bw",
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
        train_params=dict(),
        label="binary_shared_[5k,1k,500,100,50,25]bw",
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
