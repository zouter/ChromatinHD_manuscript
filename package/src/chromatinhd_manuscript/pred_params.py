import chromatinhd as chd
import chromatinhd.models.pred.model.better
import hashlib
import copy

params = []


params.append(
    dict(
        cls=chd.models.pred.model.better.Models,
        model_params=dict(
            n_frequencies=(400, 200, 100, 50, 25),
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
            distance_encoder="direct",
        ),
        train_params=dict(
            weight_decay=1e-1,
            lr=1e-4,
        ),
        label="v35",
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
            dropout_rate_fragment_embedder=0.0,
            dropout_rate_embedding2expression=0.0,
            encoder="spline_binary",
            nonlinear="silu",
            library_size_encoder="linear",
        ),
        train_params=dict(
            weight_decay=1e-1,
            lr=1e-4,
        ),
        label="v34",
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
            dropout_rate_fragment_embedder=0.0,
            dropout_rate_embedding2expression=0.0,
            encoder="radial_binary",
            nonlinear="silu",
            library_size_encoder="linear",
            distance_encoder="direct",
        ),
        train_params=dict(
            weight_decay=1e-1,
            lr=1e-4,
        ),
        label="v33",
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
            n_layers_embedding2expression=0,
            residual_embedding2expression=True,
            layernorm_embedding2expression=True,
            dropout_rate_fragment_embedder=0.0,
            dropout_rate_embedding2expression=0.0,
            encoder="radial_binary",
            nonlinear="silu",
            library_size_encoder="linear",
            distance_encoder="direct",
        ),
        train_params=dict(
            weight_decay=1e-1,
            # lr=1e-4,
            lr=1e-3,
        ),
        label="v33_additive",
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
            dropout_rate_fragment_embedder=0.0,
            dropout_rate_embedding2expression=0.0,
            encoder="radial_binary",
            nonlinear="silu",
        ),
        train_params=dict(
            weight_decay=1e-1,
            lr=1e-4,
        ),
        label="v32",
    ),
)


params.append(
    dict(
        cls=chd.models.pred.model.better.Models,
        model_params=dict(
            n_frequencies=(1000, 500, 250, 125, 63, 31),
            n_embedding_dimensions=100,
            n_layers_fragment_embedder=5,
            residual_fragment_embedder=False,
            n_layers_embedding2expression=5,
            residual_embedding2expression=False,
            dropout_rate_fragment_embedder=0.0,
            dropout_rate_embedding2expression=0.0,
            encoder="radial_binary",
            distance_encoder="split",
            library_size_encoder="linear",
        ),
        train_params=dict(
            weight_decay=1e-1,
        ),
        label="v31",
    ),
)


params.append(
    dict(
        cls=chd.models.pred.model.better.Models,
        model_params=dict(
            n_frequencies=(1000, 500, 250, 125, 63, 31),
            n_embedding_dimensions=100,
            n_layers_fragment_embedder=5,
            residual_fragment_embedder=False,
            n_layers_embedding2expression=5,
            residual_embedding2expression=False,
            dropout_rate_fragment_embedder=0.0,
            dropout_rate_embedding2expression=0.0,
            encoder="radial_binary",
            distance_encoder="split",
        ),
        train_params=dict(
            weight_decay=1e-1,
        ),
        label="v30",
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
