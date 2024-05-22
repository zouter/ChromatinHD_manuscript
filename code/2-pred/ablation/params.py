import chromatinhd as chd
import chromatinhd.models.pred.model.better
import hashlib
import copy
import pandas as pd
import itertools

main_param_ids = []

params = []

v33_params = dict(
    cls=chd.models.pred.model.better.Models,
    model_params=dict(
        n_frequencies=(1000, 500, 250, 125, 63, 31),
        n_embedding_dimensions=100,
        n_layers_fragment_embedder=1,
        residual_fragment_embedder=True,
        n_layers_embedding2expression=5,
        residual_embedding2expression=True,
        layernorm_embedding2expression=True,
        dropout_rate_fragment_embedder=0.0,
        dropout_rate_embedding2expression=0.0,
        encoder="radial_binary",
        nonlinear="silu",
        library_size_encoder="linear",
        library_size_encoder_kwargs=dict(scale=0.5),
        distance_encoder="direct",
    ),
    train_params=dict(
        weight_decay=1e-1,
        lr=1e-4,
    ),
    label="v33",
)
p = copy.deepcopy(v33_params)
p["interpret"] = True
main_param_ids.append("v33")
params.append(p)

# v33_cpu
p = copy.deepcopy(v33_params)
p["model_params"]["library_size_encoder"] = None
p["model_params"]["n_frequencies"] = (200, 100, 50, 25)
p["train_params"]["device"] = "cpu"
p["interpretation_params"] = dict(
    device="cpu",
    window_sizes=(50,),
    relative_stride=1.0,
)
p["interpret"] = True
p["label"] = "v33_cpu"
params.append(p)
main_param_ids.append("v33_cpu")

# v33_windows50_spline_binary
p = copy.deepcopy(v33_params)
p["train_params"]["device"] = "cuda:0"
p["model_params"]["n_frequencies"] = (1000, 500, 250, 125, 63, 31, 15)
p["model_params"]["encoder"] = "spline_binary"
p["interpretation_params"] = dict(
    window_sizes=(50,),
    relative_stride=1,
)
p["interpret"] = True
p["label"] = "v33_windows50_spline_binary"
params.append(p)
main_param_ids.append("v33_windows50_spline_binary")

# v33_windows50
p = copy.deepcopy(v33_params)
p["train_params"]["device"] = "cuda:0"
p["model_params"]["n_frequencies"] = (200, 100, 50, 25)
p["interpretation_params"] = dict(
    window_sizes=(50,),
    relative_stride=1,
)
p["interpret"] = True
p["label"] = "v33_windows50"
params.append(p)
main_param_ids.append("v33_windows50")

# v33_windows100
p = copy.deepcopy(v33_params)
p["train_params"]["device"] = "cuda:0"
p["model_params"]["n_frequencies"] = (200, 100, 50, 25)
p["interpretation_params"] = dict(
    window_sizes=(100,),
    relative_stride=1,
)
p["interpret"] = True
p["label"] = "v33_windows100"
params.append(p)
main_param_ids.append("v33_windows100")

# v33_windows500
p = copy.deepcopy(v33_params)
p["train_params"]["device"] = "cuda:0"
p["model_params"]["n_frequencies"] = (200, 100, 50, 25)
p["interpretation_params"] = dict(
    window_sizes=(500,),
    relative_stride=1,
)
p["interpret"] = True
p["label"] = "v33_windows500"
params.append(p)
main_param_ids.append("v33_windows500")

# v33_rtx4060
p = copy.deepcopy(v33_params)
p["train_params"]["device"] = "cuda:1"
p["train_params"]["n_cells_step"] = 200
p["interpretation_params"] = dict(
    device="cuda:1",
    window_sizes=(50,),
    relative_stride=1.0,
)
p["interpret"] = True
p["label"] = "v33_rtx4060"
params.append(p)
main_param_ids.append("v33_rtx4060")

# v33_nodistance
p = copy.deepcopy(v33_params)
p["model_params"]["distance_encoder"] = None
p["label"] = "v33_nodistance"
params.append(p)
main_param_ids.append("v33_nodistance")

# v33_nolib
p = copy.deepcopy(v33_params)
p["model_params"]["library_size_encoder"] = None
p["label"] = "v33_nolib"
params.append(p)
main_param_ids.append("v33_nolib")


nonlinear_design = []

# v33_relu
p = copy.deepcopy(v33_params)
p["model_params"]["nonlinear"] = "relu"
p["label"] = "v33_relu"
params.append(p)
main_param_ids.append("v33_relu")
nonlinear_design.append(
    {
        "relu": True,
        "label": "v33_relu",
    }
)

# v33_gelu
p = copy.deepcopy(v33_params)
p["model_params"]["nonlinear"] = "gelu"
p["label"] = "v33_gelu"
params.append(p)
main_param_ids.append("v33_gelu")
nonlinear_design.append(
    {
        "gelu": True,
        "label": "v33_gelu",
    }
)

# v33_sigmoid
p = copy.deepcopy(v33_params)
p["model_params"]["nonlinear"] = "sigmoid"
p["label"] = "v33_sigmoid"
params.append(p)
main_param_ids.append("v33_sigmoid")
nonlinear_design.append(
    {
        "sigmoid": True,
        "label": "v33_sigmoid",
    }
)

# v33_tanh
p = copy.deepcopy(v33_params)
p["model_params"]["nonlinear"] = "tanh"
p["label"] = "v33_tanh"
params.append(p)
main_param_ids.append("v33_tanh")
nonlinear_design.append(
    {
        "tanh": True,
        "label": "v33_tanh",
    }
)


# v33_silu
p = copy.deepcopy(v33_params)
p["model_params"]["nonlinear"] = "silu"
p["label"] = "v33_silu"
params.append(p)
main_param_ids.append("v33_silu")
nonlinear_design.append(
    {
        "silu": True,
        "label": "v33_silu",
    }
)

nonlinear_design = pd.DataFrame.from_records(nonlinear_design).set_index("label")

## lr
lr_design = []
for lr in [5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5]:
    p = copy.deepcopy(v33_params)
    p["train_params"]["lr"] = lr
    p["label"] = f"v33_lr{lr}"
    params.append(p)
    lr_design.append(
        {
            "lr": lr,
            "label": p["label"],
        }
    )
    main_param_ids.append(p["label"])
lr_design = pd.DataFrame.from_records(lr_design).set_index("label")

## wd
wd_design = []
for wd in [5e-1, 2e-1, 1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3]:
    p = copy.deepcopy(v33_params)
    p["train_params"]["weight_decay"] = wd
    p["label"] = f"v33_wd{wd}"
    params.append(p)
    wd_design.append(
        {
            "wd": wd,
            "label": p["label"],
        }
    )
    main_param_ids.append(p["label"])
wd_design = pd.DataFrame.from_records(wd_design).set_index("label")

## early stopping
p = copy.deepcopy(v33_params)
p["train_params"]["early_stopping"] = False
p["label"] = "v33_noearlystopping"
params.append(p)
main_param_ids.append(p["label"])

## # nhidden
nhidden_design = []
for nhidden in [1, 2, 5, 10, 20, 50, 100, 200, 500]:
    p = copy.deepcopy(v33_params)
    p["model_params"]["n_embedding_dimensions"] = nhidden
    p["label"] = f"v33_nhidden{nhidden}"
    params.append(p)
    nhidden_design.append(
        {
            "nhidden": nhidden,
            "label": p["label"],
        }
    )
    main_param_ids.append(p["label"])
nhidden_design = pd.DataFrame.from_records(nhidden_design).set_index("label")


from itertools import combinations


def all_combinations(elements):
    n = len(elements)
    all_comb = []
    for k in range(1, n + 1):
        for comb in combinations(elements, k):
            all_comb.append(comb)
    return all_comb


def progressive_combinations(elements):
    n = len(elements)
    all_comb = []
    for i in range(n - 1):
        all_comb.append(elements[i:])
    return all_comb


def regressive_combinations(elements):
    n = len(elements)
    all_comb = []
    for i in range(n):
        all_comb.append(elements[: n - i])
    return all_comb


# N frequencies
options = [1000, 500, 250, 125, 63, 31, 15]

nfrequencies_design = []
# for n in all_combinations(options):
# for n in progressive_combinations(options):
for n in regressive_combinations(options):
    p = copy.deepcopy(v33_params)
    p["model_params"]["n_frequencies"] = n
    p["label"] = f"v33_{n}frequencies"
    params.append(p)
    nfrequencies_design.append(
        {
            "n_frequencies": n,
            "label": p["label"],
        }
    )
    main_param_ids.append(p["label"])
nfrequencies_design = pd.DataFrame.from_records(nfrequencies_design).set_index("label")


options = [6250, 3125, 1563, 781, 391, 195, 98, 49, 25, 13]

nfrequencies_tophat_design = []
# for n in all_combinations(options):
# for n in progressive_combinations(options):
for n in regressive_combinations(options):
    p = copy.deepcopy(v33_params)
    p["model_params"]["n_frequencies"] = n
    p["model_params"]["encoder"] = "tophat_binary"
    p["label"] = f"v33_tophat_{n}frequencies"
    params.append(p)
    nfrequencies_tophat_design.append(
        {
            "n_frequencies": n,
            "label": p["label"],
        }
    )
    main_param_ids.append(p["label"])
nfrequencies_tophat_design = pd.DataFrame.from_records(nfrequencies_tophat_design).set_index("label")

## n layers
nlayers_design = []
for layerfe, layere2e in itertools.product([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]):
    p = copy.deepcopy(v33_params)
    p["model_params"]["n_layers_fragment_embedder"] = layerfe
    p["model_params"]["n_layers_embedding2expression"] = layere2e
    p["label"] = f"v33_{layerfe}layerfe_{layere2e}layere2e"
    params.append(p)
    nlayers_design.append(
        {
            "layerfe": layerfe,
            "layere2e": layere2e,
            "label": p["label"],
        }
    )
    main_param_ids.append(p["label"])
nlayers_design = pd.DataFrame.from_records(nlayers_design).set_index("label")


## n cells
ncells_design = []
for ncells in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]:
    p = copy.deepcopy(v33_params)
    p["train_params"]["n_cells_train"] = ncells
    p["label"] = f"v33_{ncells}ncellstrain"
    params.append(p)
    ncells_design.append(
        {
            "n_cells_train": ncells,
            "label": p["label"],
        }
    )
    main_param_ids.append(p["label"])
ncells_design = pd.DataFrame.from_records(ncells_design).set_index("label")

# Residual
residual_design = []

p = copy.deepcopy(v33_params)
p["model_params"]["residual_fragment_embedder"] = False
p["model_params"]["residual_embedding2expression"] = False
p["label"] = "v33_no_residual"
params.append(p)
main_param_ids.append(p["label"])
residual_design.append(
    {
        "residual_fragment_embedder": False,
        "residual_embedding2expression": False,
        "label": p["label"],
    }
)

p = copy.deepcopy(v33_params)
p["model_params"]["residual_fragment_embedder"] = True
p["model_params"]["residual_embedding2expression"] = False
p["label"] = "v33_residualfe"
params.append(p)
main_param_ids.append(p["label"])
residual_design.append(
    {
        "residual_fragment_embedder": True,
        "residual_embedding2expression": False,
        "label": p["label"],
    }
)

p = copy.deepcopy(v33_params)
p["model_params"]["residual_fragment_embedder"] = False
p["model_params"]["residual_embedding2expression"] = True
p["label"] = "v33_residual2e"
params.append(p)
main_param_ids.append(p["label"])
residual_design.append(
    {
        "residual_fragment_embedder": False,
        "residual_embedding2expression": True,
        "label": p["label"],
    }
)

p = copy.deepcopy(v33_params)
p["model_params"]["residual_fragment_embedder"] = True
p["model_params"]["residual_embedding2expression"] = True
p["label"] = "v33_residualfull"
params.append(p)
main_param_ids.append(p["label"])
residual_design.append(
    {
        "residual_fragment_embedder": True,
        "residual_embedding2expression": True,
        "label": p["label"],
    }
)

residual_design = pd.DataFrame.from_records(residual_design).set_index("label")

## Dropout
dropout_design = []
for rate in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    p = copy.deepcopy(v33_params)
    p["model_params"]["dropout_rate_fragment_embedder"] = rate
    p["model_params"]["dropout_rate_embedding2expression"] = rate
    p["label"] = f"v33_dropout{rate}"
    params.append(p)
    main_param_ids.append(p["label"])

    dropout_design.append(
        {
            "rate": rate,
            "label": p["label"],
        }
    )
dropout_design = pd.DataFrame.from_records(dropout_design).set_index("label")

## Batch/layer norm
layernorm_design = []

p = copy.deepcopy(v33_params)
p["model_params"]["layernorm_fragment_embedder"] = False
p["model_params"]["layernorm_embedding2expression"] = False
p["label"] = "v33_no_layernorm"
params.append(p)
main_param_ids.append(p["label"])
layernorm_design.append(
    {
        "layernorm_fragment_embedder": False,
        "layernorm_embedding2expression": False,
        "label": p["label"],
    }
)

p = copy.deepcopy(v33_params)
p["model_params"]["layernorm_fragment_embedder"] = False
p["model_params"]["layernorm_embedding2expression"] = False
p["model_params"]["batchnorm_fragment_embedder"] = True
p["model_params"]["batchnorm_embedding2expression"] = False
p["label"] = "v33_batchnormfe"
params.append(p)
main_param_ids.append(p["label"])
layernorm_design.append(
    {
        "layernorm_fragment_embedder": False,
        "layernorm_embedding2expression": False,
        "batchnorm_fragment_embedder": True,
        "batchnorm_embedding2expression": False,
        "label": p["label"],
    }
)


p = copy.deepcopy(v33_params)
p["model_params"]["layernorm_fragment_embedder"] = False
p["model_params"]["layernorm_embedding2expression"] = False
p["model_params"]["batchnorm_fragment_embedder"] = False
p["model_params"]["batchnorm_embedding2expression"] = True
p["label"] = "v33_batchnorm2e"
params.append(p)
main_param_ids.append(p["label"])
layernorm_design.append(
    {
        "layernorm_fragment_embedder": False,
        "layernorm_embedding2expression": False,
        "batchnorm_fragment_embedder": False,
        "batchnorm_embedding2expression": True,
        "label": p["label"],
    }
)


p = copy.deepcopy(v33_params)
p["model_params"]["layernorm_fragment_embedder"] = False
p["model_params"]["layernorm_embedding2expression"] = False
p["model_params"]["batchnorm_fragment_embedder"] = True
p["model_params"]["batchnorm_embedding2expression"] = True
p["label"] = "v33_batchnormfull"
params.append(p)
main_param_ids.append(p["label"])
layernorm_design.append(
    {
        "layernorm_fragment_embedder": False,
        "layernorm_embedding2expression": False,
        "batchnorm_fragment_embedder": True,
        "batchnorm_embedding2expression": True,
        "label": p["label"],
    }
)


p = copy.deepcopy(v33_params)
p["model_params"]["layernorm_fragment_embedder"] = True
p["model_params"]["layernorm_embedding2expression"] = False
p["label"] = "v33_layernormfe"
params.append(p)
main_param_ids.append(p["label"])
layernorm_design.append(
    {
        "layernorm_fragment_embedder": True,
        "layernorm_embedding2expression": False,
        "label": p["label"],
    }
)

p = copy.deepcopy(v33_params)
p["model_params"]["layernorm_fragment_embedder"] = False
p["model_params"]["layernorm_embedding2expression"] = True
p["label"] = "v33_layernorm2e"
params.append(p)
main_param_ids.append(p["label"])
layernorm_design.append(
    {
        "layernorm_fragment_embedder": False,
        "layernorm_embedding2expression": True,
        "label": p["label"],
    }
)

p = copy.deepcopy(v33_params)
p["model_params"]["layernorm_fragment_embedder"] = True
p["model_params"]["layernorm_embedding2expression"] = True
p["label"] = "v33_layernormfull"
params.append(p)
main_param_ids.append(p["label"])
layernorm_design.append(
    {
        "layernorm_fragment_embedder": True,
        "layernorm_embedding2expression": True,
        "label": p["label"],
    }
)

layernorm_design = pd.DataFrame.from_records(layernorm_design).set_index("label")

## Counter
params.append(
    dict(
        cls=chd.models.pred.model.better.Models,
        model_params=dict(
            n_embedding_dimensions=1,
            n_layers_fragment_embedder=1,
            n_layers_embedding2expression=1,
            encoder="nothing",
        ),
        train_params=dict(
            weight_decay=1e-1,
            lr=1e-2,
        ),
        label="counter",
    ),
)
main_param_ids.append("counter")


## Positional encodings
encoder_design = []
encoders = [
    ["direct", {}, "direct"],
    ["sine", {"n_frequencies": 20}, "sine"],
    # ["exponential", {}, "exponential"],
    # ["tophat", dict(n_frequencies=1000), "tophat"],
    # ["radial", dict(n_frequencies=1000), "radial"],
    ["tophat_binary", dict(n_frequencies=(1000, 500, 250, 125, 63, 31)), "tophat_binary"],
    ["radial_binary", dict(n_frequencies=(1000, 500, 250, 125, 63, 31)), "radial_binary"],
    # ["radial_binary", dict(n_frequencies=(1000, 750, 500, 375, 250, 180, 125, 63, 31)), "radial_binary4"],
    ["spline_binary", dict(n_frequencies=(1000, 500, 250, 125, 63, 31)), "spline_binary"],
]
for encoder, encoder_kwargs, label in encoders:
    p = copy.deepcopy(v33_params)
    label = f"v33_{label}"
    p["model_params"]["encoder"] = encoder
    if "n_frequencies" in encoder_kwargs:
        p["model_params"]["n_frequencies"] = encoder_kwargs["n_frequencies"]
        del encoder_kwargs["n_frequencies"]

    p["model_params"]["encoder_kwargs"] = encoder_kwargs
    p["label"] = label
    params.append(p)
    main_param_ids.append(p["label"])

    encoder_design.append(
        {
            "encoder": encoder,
            "encoder_kwargs": encoder_kwargs,
            "label": label,
        }
    )
encoder_design = pd.DataFrame.from_records(encoder_design).set_index("label")


####


params.append(
    dict(
        cls=chd.models.pred.model.better.Models,
        model_params=dict(
            n_frequencies=(1000, 500, 250, 125, 63, 31),
            n_embedding_dimensions=100,
            n_layers_fragment_embedder=5,
            residual_fragment_embedder=True,
            n_layers_embedding2expression=5,
            residual_embedding2expression=True,
            encoder="radial_binary",
            distance_encoder="split",
        ),
        train_params=dict(
            weight_decay=1e-1,
        ),
        label="radial_binary_1000-31frequencies_splitdistance_residualfull",
    ),
)


params.append(
    dict(
        cls=chd.models.pred.model.better.Models,
        model_params=dict(
            n_frequencies=(1000, 500, 250, 125, 63, 31),
            n_embedding_dimensions=100,
            n_layers_fragment_embedder=5,
            residual_fragment_embedder=True,
            n_layers_embedding2expression=5,
            residual_embedding2expression=True,
            batchnorm_embedding2expression=True,
            encoder="radial_binary",
            nonlinear="silu",
        ),
        train_params=dict(
            weight_decay=1e-1,
        ),
        label="radial_binary_1000-31frequencies_residualfull_bne2e",
    ),
)


params.append(
    dict(
        cls=chd.models.pred.model.better.Models,
        model_params=dict(
            n_frequencies=(1000, 500, 250, 125, 63, 31),
            n_embedding_dimensions=100,
            n_layers_fragment_embedder=5,
            residual_fragment_embedder=True,
            n_layers_embedding2expression=5,
            residual_embedding2expression=True,
            layernorm_embedding2expression=True,
            encoder="radial_binary",
            nonlinear="silu",
        ),
        train_params=dict(
            weight_decay=1e-1,
        ),
        label="radial_binary_1000-31frequencies_residualfull_lne2e",
    ),
)

params.append(
    dict(
        cls=chd.models.pred.model.better.Models,
        model_params=dict(
            n_frequencies=(1000, 500, 250, 125, 63, 31),
            n_embedding_dimensions=100,
            n_layers_fragment_embedder=5,
            residual_fragment_embedder=True,
            n_layers_embedding2expression=5,
            residual_embedding2expression=True,
            layernorm_embedding2expression=True,
            encoder="radial_binary",
            nonlinear="silu",
            library_size_encoder="linear",
        ),
        train_params=dict(
            weight_decay=1e-1,
        ),
        label="radial_binary_1000-31frequencies_residualfull_lne2e_linearlib",
    ),
)


params.append(
    dict(
        cls=chd.models.pred.model.better.Models,
        model_params=dict(
            n_frequencies=(1000, 500, 250, 125, 63, 31),
            n_embedding_dimensions=100,
            n_layers_fragment_embedder=1,
            residual_fragment_embedder=True,
            n_layers_embedding2expression=5,
            residual_embedding2expression=True,
            layernorm_embedding2expression=True,
            encoder="radial_binary",
            nonlinear="silu",
        ),
        train_params=dict(
            weight_decay=1e-1,
        ),
        label="radial_binary_1000-31frequencies_residualfull_lne2e_1layerfe",
    ),
)

# nlayers_design = []
# for layerfe, layere2e in itertools.product([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]):
#     params.append(
#         dict(
#             cls=chd.models.pred.model.better.Models,
#             model_params=dict(
#                 n_frequencies=(1000, 500, 250, 125, 63, 31),
#                 n_embedding_dimensions=100,
#                 n_layers_fragment_embedder=layerfe,
#                 residual_fragment_embedder=True,
#                 n_layers_embedding2expression=layere2e,
#                 residual_embedding2expression=True,
#                 layernorm_embedding2expression=True,
#                 encoder="radial_binary",
#                 nonlinear="silu",
#             ),
#             train_params=dict(
#                 weight_decay=1e-1,
#             ),
#             label=f"radial_binary_1000-31frequencies_residualfull_lne2e_{layerfe}layerfe_{layere2e}layere2e",
#         ),
#     )
#     nlayers_design.append(
#         {
#             "layerfe": layerfe,
#             "layere2e": layere2e,
#             "label": f"radial_binary_1000-31frequencies_residualfull_lne2e_{layerfe}layerfe_{layere2e}layere2e",
#         }
#     )
# nlayers_design = pd.DataFrame.from_records(nlayers_design).set_index("label")


# lr_design = []
# for lr in [5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5]:
#     params.append(
#         dict(
#             cls=chd.models.pred.model.better.Models,
#             model_params=dict(
#                 n_frequencies=(1000, 500, 250, 125, 63, 31),
#                 n_embedding_dimensions=100,
#                 n_layers_fragment_embedder=1,
#                 residual_fragment_embedder=True,
#                 n_layers_embedding2expression=5,
#                 residual_embedding2expression=True,
#                 layernorm_embedding2expression=True,
#                 encoder="radial_binary",
#                 nonlinear="silu",
#             ),
#             train_params=dict(
#                 weight_decay=1e-1,
#                 lr=lr,
#             ),
#             label=f"radial_binary_1000-31frequencies_residualfull_lne2e_lr{lr}",
#         ),
#     )
#     lr_design.append(
#         {
#             "lr": lr,
#             "label": f"radial_binary_1000-31frequencies_residualfull_lne2e_lr{lr}",
#         }
#     )
# lr_design = pd.DataFrame.from_records(lr_design).set_index("label")


# wd_design = []
# for wd in [1e-1, 1e-2, 1e-3, 1e-4]:
#     params.append(
#         dict(
#             cls=chd.models.pred.model.better.Models,
#             model_params=dict(
#                 n_frequencies=(1000, 500, 250, 125, 63, 31),
#                 n_embedding_dimensions=100,
#                 n_layers_fragment_embedder=1,
#                 residual_fragment_embedder=True,
#                 n_layers_embedding2expression=5,
#                 residual_embedding2expression=True,
#                 layernorm_embedding2expression=True,
#                 encoder="radial_binary",
#                 nonlinear="silu",
#             ),
#             train_params=dict(
#                 weight_decay=wd,
#             ),
#             label=f"radial_binary_1000-31frequencies_residualfull_lne2e_wd{wd}",
#         ),
#     )
#     wd_design.append(
#         {
#             "wd": wd,
#             "label": f"radial_binary_1000-31frequencies_residualfull_lne2e_wd{wd}",
#         }
#     )
# wd_design = pd.DataFrame.from_records(wd_design).set_index("label")


params.append(
    dict(
        cls=chd.models.pred.model.better.Models,
        model_params=dict(
            n_frequencies=(1000, 500, 250, 125, 63, 31),
            n_embedding_dimensions=100,
            n_layers_fragment_embedder=5,
            residual_fragment_embedder=True,
            layernorm_fragment_embedder=True,
            n_layers_embedding2expression=5,
            residual_embedding2expression=True,
            layernorm_embedding2expression=True,
            encoder="radial_binary",
            nonlinear="silu",
        ),
        train_params=dict(
            weight_decay=1e-1,
        ),
        label="radial_binary_1000-31frequencies_residualfull_lnfull",
    ),
)


params.append(
    dict(
        cls=chd.models.pred.model.better.Models,
        model_params=dict(
            n_frequencies=(1000, 500, 250, 125, 63, 31),
            n_embedding_dimensions=100,
            n_layers_fragment_embedder=5,
            residual_fragment_embedder=True,
            n_layers_embedding2expression=5,
            residual_embedding2expression=True,
            encoder="radial_binary",
            distance_encoder="split",
        ),
        train_params=dict(
            weight_decay=1e-1,
            optimizer="adamw",
        ),
        label="radial_binary_1000-31frequencies_splitdistance_residualfull_adamw",
    ),
)

params.append(
    dict(
        cls=chd.models.pred.model.better.Models,
        model_params=dict(
            n_embedding_dimensions=100,
            n_layers_fragment_embedder=1,
            residual_fragment_embedder=True,
            n_layers_embedding2expression=5,
            residual_embedding2expression=True,
            layernorm_embedding2expression=True,
            encoder="spline_binary",
            nonlinear="silu",
        ),
        train_params=dict(
            weight_decay=1e-1,
        ),
        label="spline_binary_residualfull_lne2e_1layerfe",
    ),
)


params.append(
    dict(
        cls=chd.models.pred.model.better.Models,
        model_params=dict(
            n_frequencies=(1000, 500, 250, 125, 63, 31),
            n_embedding_dimensions=100,
            n_layers_fragment_embedder=5,
            n_layers_embedding2expression=5,
            encoder="radial_binary",
            distance_encoder="split",
        ),
        train_params=dict(
            weight_decay=1e-1,
        ),
        label="radial_binary_1000-31frequencies_splitdistance",
    ),
)


params.append(
    dict(
        cls=chd.models.pred.model.better.Models,
        model_params=dict(
            n_frequencies=(1000, 500, 250, 125, 63, 31),
            n_embedding_dimensions=100,
            n_layers_fragment_embedder=5,
            n_layers_embedding2expression=5,
            encoder="radial_binary",
        ),
        train_params=dict(
            weight_decay=1e-1,
        ),
        label="radial_binary_1000-31frequencies",
    ),
)


params.append(
    dict(
        cls=chd.models.pred.model.better.Models,
        model_params=dict(
            n_frequencies=(1000, 500, 250, 125, 63, 31),
            n_embedding_dimensions=100,
            n_layers_fragment_embedder=5,
            n_layers_embedding2expression=5,
            encoder="radial_binary",
        ),
        train_params=dict(
            weight_decay=1e-1,
            optimizer="adamw",
        ),
        label="radial_binary_1000-31frequencies_adamw",
    ),
)


params.append(
    dict(
        cls=chd.models.pred.model.better.Models,
        model_params=dict(
            n_frequencies=(1000,),
            n_embedding_dimensions=100,
            n_layers_fragment_embedder=5,
            n_layers_embedding2expression=5,
            encoder="radial_binary",
        ),
        train_params=dict(
            weight_decay=1e-1,
        ),
        label="radial_binary_1000frequencies",
    ),
)


params.append(
    dict(
        cls=chd.models.pred.model.better.Models,
        model_params=dict(
            n_frequencies=(31,),
            n_embedding_dimensions=100,
            n_layers_fragment_embedder=5,
            n_layers_embedding2expression=5,
            encoder="radial_binary",
        ),
        train_params=dict(
            weight_decay=1e-1,
        ),
        label="radial_binary_31frequencies",
    ),
)


params.append(
    dict(
        cls=chd.models.pred.model.better.Models,
        model_params=dict(
            n_frequencies=(1000, 500, 250, 125, 63, 31),
            n_embedding_dimensions=10,
            n_layers_fragment_embedder=5,
            n_layers_embedding2expression=5,
            encoder="radial_binary",
        ),
        train_params=dict(
            weight_decay=1e-1,
        ),
        label="radial_binary_1000-31frequencies_10embedding",
    ),
)


params.append(
    dict(
        cls=chd.models.pred.model.better.Models,
        model_params=dict(
            n_frequencies=50,
            n_embedding_dimensions=100,
            n_layers_fragment_embedder=5,
            residual_fragment_embedder=True,
            n_layers_embedding2expression=5,
            residual_embedding2expression=True,
            layernorm_embedding2expression=True,
            encoder="sine",
            nonlinear="silu",
        ),
        train_params=dict(
            weight_decay=1e-1,
        ),
        label="sine_50frequencies_residualfull_lne2e",
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
