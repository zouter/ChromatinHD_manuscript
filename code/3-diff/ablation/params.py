import chromatinhd as chd
import hashlib
import copy
import pandas as pd

params = []


params.append(
    dict(
        model_params=dict(
            encoder="shared",
            encoder_params=dict(
                binwidths=(5000, 1000, 500, 100, 50, 25),
            ),
        ),
        # train_params=dict(early_stopping=False, n_epochs=150, lr=1e-3),
        train_params=dict(n_cells_step=5000, early_stopping=False, n_epochs=150, lr=1e-3),
        label="v31",
    ),
)


params.append(
    dict(
        model_params=dict(
            encoder="shared",
            encoder_params=dict(
                binwidths=(5000, 1000, 500, 100, 50, 25),
                delta_regularization=False,
            ),
        ),
        # train_params=dict(early_stopping=False, n_epochs=150, lr=1e-3),
        train_params=dict(n_cells_step=5000, early_stopping=False, n_epochs=150, lr=1e-3),
        label="v31_wdeltareg-no",
    ),
)


binwidths = (5000, 1000, 500, 200, 100, 50, 25)
binwidth_titration = []
for i in range(1, len(binwidths)):
    p = dict(
        model_params=dict(
            encoder="shared",
            encoder_params=dict(
                binwidths=binwidths[i:],
            ),
        ),
        train_params=dict(early_stopping=False, n_epochs=150, lr=1e-3),
        label=f"v31_{binwidths[i:]}bw",
    )
    params.append(p)
    binwidth_titration.append({"label": p["label"], "binwidths": p["model_params"]["encoder_params"]["binwidths"]})
binwidth_titration = pd.DataFrame(binwidth_titration)

## Binwidth combinations
from itertools import combinations


def all_combinations(elements):
    n = len(elements)
    all_comb = []
    for k in range(1, n + 1):
        for comb in combinations(elements, k):
            all_comb.append(comb)
    return all_comb


binwidths = (5000, 1000, 500, 200, 100, 50, 25)
binwidth_combinations = []
for bw in all_combinations(binwidths):
    if bw[-1] == 200:
        continue
    binwidth_combinations.append({"label": p["label"], "binwidths": bw})
    if any(p["label"] == f"v31_{bw}bw" for p in params):
        continue
    p = dict(
        model_params=dict(
            encoder="shared",
            encoder_params=dict(
                binwidths=bw,
            ),
        ),
        train_params=dict(early_stopping=False, n_epochs=150, lr=1e-3),
        label=f"v31_{bw}bw",
    )
    params.append(p)
binwidth_combinations = pd.DataFrame(binwidth_combinations)


fast_param = dict(
    model_params=dict(
        encoder="shared",
        encoder_params=dict(
            binwidths=(5000, 1000, 500, 100, 50, 25),
        ),
    ),
    train_params=dict(n_cells_step=5000, n_epochs=300, lr=1e-3, early_stopping=False),
)

w_delta_p_scales = (0.1, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 20.0)
w_delta_p_scale_design = []
for w_delta_p_scale in w_delta_p_scales:
    p = copy.deepcopy(fast_param)
    p["model_params"]["encoder_params"]["delta_p_scale"] = w_delta_p_scale
    p["label"] = f"v31a-w_delta_p_scale{w_delta_p_scale}"
    params.append(p)
    w_delta_p_scale_design.append({"w_delta_p_scale": w_delta_p_scale, "label": p["label"]})
w_delta_p_scale_design = pd.DataFrame(w_delta_p_scale_design)

###

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

# binwidths = (5000, 1000, 500, 200, 100, 50, 25)
# binwidth_titration_ids = []
# for i in range(1, len(binwidths)):
#     params.append(
#         dict(
#             model_params=dict(
#                 encoder="shared",
#                 encoder_params=dict(
#                     binwidths=binwidths[i:],
#                 ),
#             ),
#             train_params=dict(),
#             label=f"binary_{binwidths[i:]}bw",
#         ),
#     )
#     binwidth_titration_ids.append(f"binary_{binwidths[i:]}bw")


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


params.append(
    dict(
        model_params=dict(
            encoder="shared_lowrank",
            encoder_params=dict(
                binwidths=(5000, 1000, 500, 100, 50, 25),
            ),
        ),
        train_params=dict(),
        label="binary_shared_lowrank_[5k,1k,500,100,50,25]bw",
    ),
)


lrs = (1e-1, 5e-2, 1e-2, 5e-3, 1e-3)
lr_titration_ids = []
for lr in lrs:
    params.append(
        dict(
            model_params=dict(
                encoder="shared",
                encoder_params=dict(
                    binwidths=(5000, 1000, 500, 100, 50, 25),
                ),
            ),
            train_params=dict(lr=lr),
            label=f"binary_shared_[5k,1k,500,100,50,25]bw_lr{lr}",
        ),
    )
    lr_titration_ids.append(params[-1]["label"])


params.append(
    dict(
        model_params=dict(
            encoder="shared",
            encoder_params=dict(
                binwidths=(5000, 1000, 500, 100, 50, 25),
            ),
        ),
        train_params=dict(n_cells_step=5000),
        label="binary_shared_[5k,1k,500,100,50,25]bw_5000ncellsstep",
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
        train_params=dict(n_cells_step=5000, early_stopping=False),
        label="binary_shared_[5k,1k,500,100,50,25]bw_5000ncellsstep_noearlystop",
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
        train_params=dict(n_cells_step=5000, early_stopping=False, n_epochs=100),
        label="binary_shared_[5k,1k,500,100,50,25]bw_5000ncellsstep_noearlystop_100epochs",
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
        train_params=dict(n_cells_step=5000, early_stopping=False, n_epochs=500),
        label="binary_shared_[5k,1k,500,100,50,25]bw_5000ncellsstep_noearlystop_500epochs",
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
