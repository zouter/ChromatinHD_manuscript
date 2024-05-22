import chromatinhd as chd
import chromatinhd.models.pred.model.better
import hashlib
import copy
import pandas as pd
import itertools

params = []

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
        label="counter",
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

nlayers_design = []
for layerfe, layere2e in itertools.product([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]):
    params.append(
        dict(
            cls=chd.models.pred.model.better.Models,
            model_params=dict(
                n_frequencies=(1000, 500, 250, 125, 63, 31),
                n_embedding_dimensions=100,
                n_layers_fragment_embedder=layerfe,
                residual_fragment_embedder=True,
                n_layers_embedding2expression=layere2e,
                residual_embedding2expression=True,
                layernorm_embedding2expression=True,
                encoder="radial_binary",
                nonlinear="silu",
            ),
            train_params=dict(
                weight_decay=1e-1,
            ),
            label=f"radial_binary_1000-31frequencies_residualfull_lne2e_{layerfe}layerfe_{layere2e}layere2e",
        ),
    )
    nlayers_design.append(
        {
            "layerfe": layerfe,
            "layere2e": layere2e,
            "label": f"radial_binary_1000-31frequencies_residualfull_lne2e_{layerfe}layerfe_{layere2e}layere2e",
        }
    )
nlayers_design = pd.DataFrame.from_records(nlayers_design).set_index("label")


lr_design = []
for lr in [5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5]:
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
                lr=lr,
            ),
            label=f"radial_binary_1000-31frequencies_residualfull_lne2e_lr{lr}",
        ),
    )
    lr_design.append(
        {
            "lr": lr,
            "label": f"radial_binary_1000-31frequencies_residualfull_lne2e_lr{lr}",
        }
    )
lr_design = pd.DataFrame.from_records(lr_design).set_index("label")


wd_design = []
for wd in [1e-1, 1e-2, 1e-3, 1e-4]:
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
                weight_decay=wd,
            ),
            label=f"radial_binary_1000-31frequencies_residualfull_lne2e_wd{wd}",
        ),
    )
    wd_design.append(
        {
            "wd": wd,
            "label": f"radial_binary_1000-31frequencies_residualfull_lne2e_wd{wd}",
        }
    )
wd_design = pd.DataFrame.from_records(wd_design).set_index("label")


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
