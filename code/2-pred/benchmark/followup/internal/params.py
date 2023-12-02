params = []

import chromatinhd as chd
import chromatinhd.models.pred.model.better
import hashlib

# params.append(
#     dict(
#         cls=chd.models.pred.model.better.Model,
#         n_frequencies=100,
#         n_embedding_dimensions=100,
#         n_layers_fragment_embedder=5,
#         n_layers_embedding2expression=5,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="tophat",
#         label="tophat",
#     ),
# )
params.append(
    dict(
        cls=chd.models.pred.model.better.Model,
        n_frequencies=100,
        n_embedding_dimensions=100,
        n_layers_fragment_embedder=5,
        n_layers_embedding2expression=5,
        dropout_rate_fragment_embedder=0.0,
        dropout_rate_embedding2expression=0.0,
        encoder="sine",
        label="sine",
    ),
)
# params.append(
#     dict(
#         cls=chd.models.pred.model.better.Model,
#         n_frequencies=100,
#         n_embedding_dimensions=20,
#         n_layers_fragment_embedder=3,
#         n_layers_embedding2expression=3,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="sine",
#         label="sine_test",
#     ),
# )
# params.append(
#     dict(
#         cls=chd.models.pred.model.better.Model,
#         n_frequencies=100,
#         n_embedding_dimensions=100,
#         n_layers_fragment_embedder=5,
#         n_layers_embedding2expression=5,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="sine",
#         distance_encoder="direct",
#         label="sine_directdistance",
#     ),
# )
params.append(
    dict(
        cls=chd.models.pred.model.better.Model,
        n_frequencies=100,
        n_embedding_dimensions=100,
        n_layers_fragment_embedder=5,
        n_layers_embedding2expression=5,
        dropout_rate_fragment_embedder=0.0,
        dropout_rate_embedding2expression=0.0,
        encoder="sine",
        distance_encoder="split",
        label="sine_splitdistance",
    ),
)
# params.append(
#     dict(
#         cls=chd.models.pred.model.better.Model,
#         n_frequencies=100,
#         n_embedding_dimensions=100,
#         n_layers_fragment_embedder=1,
#         n_layers_embedding2expression=1,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="sine",
#         label="sine_1layer",
#     ),
# )
# params.append(
#     dict(
#         cls=chd.models.pred.model.better.Model,
#         n_frequencies=100,
#         n_embedding_dimensions=100,
#         n_layers_fragment_embedder=1,
#         n_layers_embedding2expression=1,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="sine",
#         label="sine_1layer_lr1",
#         lr=1e-1,
#     ),
# )
# params.append(
#     dict(
#         cls=chd.models.pred.model.better.Model,
#         n_frequencies=100,
#         n_embedding_dimensions=100,
#         n_layers_fragment_embedder=1,
#         n_layers_embedding2expression=1,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="sine",
#         label="sine_1layer_lr2",
#         lr=1e-2,
#     ),
# )
params.append(
    dict(
        cls=chd.models.pred.model.better.Model,
        n_frequencies=100,
        n_embedding_dimensions=100,
        n_layers_fragment_embedder=5,
        n_layers_embedding2expression=5,
        dropout_rate_fragment_embedder=0.0,
        dropout_rate_embedding2expression=0.0,
        encoder="radial",
        label="radial",
    ),
)
# params.append(
#     dict(
#         cls=chd.models.pred.model.better.Model,
#         n_frequencies=100,
#         n_embedding_dimensions=100,
#         n_layers_fragment_embedder=5,
#         n_layers_embedding2expression=5,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="radial_binary",
#         label="radial_binary",
#     ),
# )
# params.append(
#     dict(
#         cls=chd.models.pred.model.better.Model,
#         n_frequencies=100,
#         n_embedding_dimensions=100,
#         n_layers_fragment_embedder=5,
#         n_layers_embedding2expression=5,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="radial_binary",
#         distance_encoder="split",
#         label="radial_binary_splitdistance",
#     ),
# )
# params.append(
#     dict(
#         cls=chd.models.pred.model.better.Model,
#         n_frequencies=100,
#         n_embedding_dimensions=100,
#         n_layers_fragment_embedder=5,
#         n_layers_embedding2expression=5,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="radial",
#         distance_encoder="split",
#         label="radial_splitdistance",
#     ),
# )
# params.append(
#     dict(
#         cls=chd.models.pred.model.better.Model,
#         n_frequencies=100,
#         n_embedding_dimensions=100,
#         n_layers_fragment_embedder=5,
#         n_layers_embedding2expression=5,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="sine2",
#         label="sine2",
#     ),
# )
# params.append(
#     dict(
#         cls=chd.models.pred.model.better.Model,
#         n_frequencies=100,
#         n_embedding_dimensions=100,
#         n_layers_fragment_embedder=5,
#         n_layers_embedding2expression=5,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="sine3",
#         label="sine3",
#     ),
# )
# params.append(
#     dict(
#         cls=chd.models.pred.model.better.Model,
#         n_frequencies=100,
#         n_embedding_dimensions=100,
#         n_layers_fragment_embedder=5,
#         n_layers_embedding2expression=5,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="direct",
#         label="direct",
#     ),
# )
# params.append(
#     dict(
#         cls=chd.models.pred.model.better.Model,
#         n_frequencies=100,
#         n_embedding_dimensions=100,
#         n_layers_fragment_embedder=5,
#         residual_fragment_embedder=True,
#         n_layers_embedding2expression=5,
#         residual_embedding2expression=True,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="radial",
#         label="radial_residual",
#     ),
# )
# params.append(
#     dict(
#         cls=chd.models.pred.model.better.Model,
#         n_frequencies=1000,
#         n_embedding_dimensions=100,
#         n_layers_fragment_embedder=5,
#         residual_fragment_embedder=False,
#         n_layers_embedding2expression=5,
#         residual_embedding2expression=False,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="radial",
#         label="radial_1000frequencies",
#     ),
# )
params.append(
    dict(
        cls=chd.models.pred.model.better.Model,
        n_frequencies=1000,
        n_embedding_dimensions=100,
        n_layers_fragment_embedder=5,
        residual_fragment_embedder=False,
        n_layers_embedding2expression=5,
        residual_embedding2expression=False,
        dropout_rate_fragment_embedder=0.0,
        dropout_rate_embedding2expression=0.0,
        encoder="radial",
        distance_encoder="split",
        label="radial_1000frequencies_splitdistance",
    ),
)
params.append(
    dict(
        cls=chd.models.pred.model.better.Model,
        n_frequencies=(1000, 500, 250, 125, 63, 31),
        n_embedding_dimensions=100,
        n_layers_fragment_embedder=5,
        residual_fragment_embedder=False,
        n_layers_embedding2expression=5,
        residual_embedding2expression=False,
        dropout_rate_fragment_embedder=0.0,
        dropout_rate_embedding2expression=0.0,
        encoder="radial_binary",
        label="radial_binary_1000-31frequencies",
    ),
)
params.append(
    dict(
        cls=chd.models.pred.model.better.Model,
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
        label="radial_binary_1000-31frequencies_splitdistance",
    ),
)
params.append(
    dict(
        cls=chd.models.pred.model.better.Model,
        n_frequencies=(1000, 500, 250, 125, 63, 31),
        n_embedding_dimensions=10,
        n_layers_fragment_embedder=5,
        residual_fragment_embedder=False,
        n_layers_embedding2expression=5,
        residual_embedding2expression=False,
        dropout_rate_fragment_embedder=0.0,
        dropout_rate_embedding2expression=0.0,
        encoder="radial_binary",
        distance_encoder="split",
        lr=1e-3,
        label="radial_binary_1000-31frequencies_10embedding_splitdistance",
    ),
)
params.append(
    dict(
        cls=chd.models.pred.model.better.Model,
        n_embedding_dimensions=100,
        n_layers_fragment_embedder=5,
        residual_fragment_embedder=False,
        n_layers_embedding2expression=5,
        residual_embedding2expression=False,
        dropout_rate_fragment_embedder=0.0,
        dropout_rate_embedding2expression=0.0,
        encoder="spline_binary",
        distance_encoder="split",
        label="spline_binary_1000-31frequencies_splitdistance",
    ),
)
# params.append(
#     dict(
#         cls=chd.models.pred.model.better.Model,
#         n_embedding_dimensions=100,
#         n_layers_fragment_embedder=5,
#         residual_fragment_embedder=False,
#         n_layers_embedding2expression=5,
#         residual_embedding2expression=False,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="spline_binary",
#         distance_encoder="split",
#         label="spline_binary_1000-31frequencies_splitdistance_wd1e-1",
#         weight_decay=1e-1,
#     ),
# )
params.append(
    dict(
        cls=chd.models.pred.model.better.Model,
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
        fragment_embedder_kwargs=dict(attention=True),
        label="radial_binary_1000-31frequencies_splitdistance_attn",
    ),
)
params.append(
    dict(
        cls=chd.models.pred.model.better.Model,
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
        fragment_embedder_kwargs=dict(attention=dict(initialization="eye")),
        label="radial_binary_1000-31frequencies_splitdistance_attn_initeye",
    ),
)


# params.append(
#     dict(
#         cls=chd.models.pred.model.better.Model,
#         n_frequencies=(1000, 500, 250, 125, 63, 31),
#         n_embedding_dimensions=100,
#         n_layers_fragment_embedder=5,
#         residual_fragment_embedder=False,
#         n_layers_embedding2expression=5,
#         residual_embedding2expression=False,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="radial_binary",
#         distance_encoder="split",
#         label="radial_binary_1000-31frequencies_splitdistance_wd1e-3",
#         weight_decay=1e-3,
#     ),
# )

params.append(
    dict(
        cls=chd.models.pred.model.better.Model,
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
        label="radial_binary_1000-31frequencies_splitdistance_wd1e-1",
        weight_decay=1e-1,
    ),
)


# params.append(
#     dict(
#         cls=chd.models.pred.model.better.Model,
#         n_frequencies=(1000, 500, 250, 125, 63, 31),
#         n_embedding_dimensions=100,
#         n_layers_fragment_embedder=5,
#         residual_fragment_embedder=False,
#         n_layers_embedding2expression=5,
#         residual_embedding2expression=False,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="radial_binary",
#         distance_encoder="split",
#         label="radial_binary_1000-31frequencies_splitdistance_wd2e-1",
#         weight_decay=2e-1,
#         n_epochs=2000,
#     ),
# )


# params.append(
#     dict(
#         cls=chd.models.pred.model.better.Model,
#         n_frequencies=(1000, 500, 250, 125, 63, 31),
#         n_embedding_dimensions=100,
#         n_layers_fragment_embedder=5,
#         residual_fragment_embedder=False,
#         n_layers_embedding2expression=5,
#         residual_embedding2expression=False,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="radial_binary",
#         distance_encoder="split",
#         label="radial_binary_1000-31frequencies_splitdistance_wd5e-1",
#         weight_decay=5e-1,
#         n_epochs=5000,
#     ),
# )


params.append(
    dict(
        cls=chd.models.pred.model.better.Model,
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
        label="radial_binary_1000-31frequencies_splitdistance_lr1e-3",
        weight_decay=1e-1,
        lr=1e-3,
        n_epochs=5000,
    ),
)


params.append(
    dict(
        cls=chd.models.pred.model.better.Model,
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
        nonlinear="silu",
        label="radial_binary_1000-31frequencies_splitdistance_silu",
    ),
)
params.append(
    dict(
        cls=chd.models.pred.model.better.Model,
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
        nonlinear="relu",
        label="radial_binary_1000-31frequencies_splitdistance_relu",
    ),
)


params.append(
    dict(
        cls=chd.models.pred.model.better.Model,
        n_frequencies=(1000, 500, 250, 125, 63, 31),
        n_embedding_dimensions=100,
        n_layers_fragment_embedder=5,
        residual_fragment_embedder=False,
        n_layers_embedding2expression=5,
        residual_embedding2expression=False,
        dropout_rate_fragment_embedder=0.0,
        dropout_rate_embedding2expression=0.0,
        encoder="radial_binary_center",
        distance_encoder="linear",
        label="radial_binary_center_1000-31frequencies_lineardistance_wd1e-1",
        weight_decay=1e-1,
    ),
)


params.append(
    dict(
        cls=chd.models.pred.model.better.Model,
        n_frequencies=(1000, 500, 250, 125, 63, 31),
        n_embedding_dimensions=100,
        n_layers_fragment_embedder=5,
        residual_fragment_embedder=False,
        n_layers_embedding2expression=5,
        residual_embedding2expression=False,
        dropout_rate_fragment_embedder=0.0,
        dropout_rate_embedding2expression=0.0,
        encoder="radial_binary",
        distance_encoder="linear",
        label="radial_binary_1000-31frequencies_lineardistance_wd1e-1",
        weight_decay=1e-1,
    ),
)


# params.append(
#     dict(
#         cls=chd.models.pred.model.better.Model,
#         n_frequencies=(4000, 2000, 1000, 500, 250, 125, 63, 31),
#         n_embedding_dimensions=100,
#         n_layers_fragment_embedder=5,
#         residual_fragment_embedder=False,
#         n_layers_embedding2expression=5,
#         residual_embedding2expression=False,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="radial_binary",
#         distance_encoder="split",
#         label="radial_binary_4000-31frequencies_splitdistance_wd1e-1",
#         weight_decay=1e-1,
#     ),
# )


# params.append(
#     dict(
#         cls=chd.models.pred.model.better.Model,
#         n_frequencies=(1000, 500, 250, 125, 63, 31),
#         n_embedding_dimensions=100,
#         n_layers_fragment_embedder=5,
#         residual_fragment_embedder=False,
#         n_layers_embedding2expression=5,
#         residual_embedding2expression=False,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="linear_binary",
#         distance_encoder="split",
#         label="linear_binary_1000-31frequencies_splitdistance_wd1e-1",
#         weight_decay=1e-1,
#     ),
# )


# params.append(
#     dict(
#         cls=chd.models.pred.model.better.Model,
#         n_frequencies=(1000, 500, 250, 125, 63, 31),
#         n_embedding_dimensions=100,
#         n_layers_fragment_embedder=5,
#         residual_fragment_embedder=False,
#         n_layers_embedding2expression=5,
#         residual_embedding2expression=False,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="radial_binary",
#         distance_encoder="split",
#         fragment_embedder_kwargs=dict(layernorm=True),
#         label="radial_binary_1000-31frequencies_splitdistance_layernormfe",
#     ),
# )
# params.append(
#     dict(
#         cls=chd.models.pred.model.better.Model,
#         n_frequencies=(1000, 500, 250, 125, 63, 31),
#         n_embedding_dimensions=100,
#         n_layers_fragment_embedder=5,
#         residual_fragment_embedder=False,
#         n_layers_embedding2expression=5,
#         residual_embedding2expression=False,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="radial_binary",
#         distance_encoder="direct",
#         label="radial_binary_1000-31frequencies_directdistance",
#     ),
# )
# params.append(
#     dict(
#         cls=chd.models.pred.model.better.Model,
#         n_frequencies=(1000, 500, 250, 125, 63, 31),
#         n_embedding_dimensions=100,
#         n_layers_fragment_embedder=1,
#         residual_fragment_embedder=False,
#         n_layers_embedding2expression=1,
#         residual_embedding2expression=False,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="radial_binary",
#         distance_encoder="direct",
#         label="radial_binary_1000-31frequencies_1-1layers_directdistance",
#     ),
# )
# params.append(
#     dict(
#         cls=chd.models.pred.model.better.Model,
#         n_frequencies=(1000, 500, 250, 125, 63, 31),
#         n_embedding_dimensions=100,
#         n_layers_fragment_embedder=5,
#         residual_fragment_embedder=False,
#         n_layers_embedding2expression=1,
#         residual_embedding2expression=False,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="radial_binary",
#         distance_encoder="direct",
#         label="radial_binary_1000-31frequencies_5-1layers_directdistance",
#     ),
# )
# params.append(
#     dict(
#         cls=chd.models.pred.model.better.Model,
#         n_frequencies=(1000, 500, 250, 125, 63, 31),
#         n_embedding_dimensions=100,
#         n_layers_fragment_embedder=1,
#         residual_fragment_embedder=False,
#         n_layers_embedding2expression=5,
#         residual_embedding2expression=False,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="radial_binary",
#         distance_encoder="direct",
#         label="radial_binary_1000-31frequencies_1-5layers_directdistance",
#     ),
# )
# params.append(
#     dict(
#         cls=chd.models.pred.model.better.Model,
#         n_frequencies=(1000, 500, 250, 125, 63, 31),
#         n_embedding_dimensions=100,
#         n_layers_fragment_embedder=0,
#         residual_fragment_embedder=False,
#         n_layers_embedding2expression=5,
#         residual_embedding2expression=False,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="radial_binary",
#         distance_encoder="direct",
#         label="radial_binary_1000-31frequencies_0-5layers_directdistance",
#     ),
# )
# params.append(
#     dict(
#         cls=chd.models.pred.model.better.Model,
#         n_frequencies=(1000, 500, 250, 125, 63, 31),
#         n_embedding_dimensions=100,
#         n_layers_fragment_embedder=10,
#         residual_fragment_embedder=False,
#         n_layers_embedding2expression=1,
#         residual_embedding2expression=False,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="radial_binary",
#         distance_encoder="direct",
#         label="radial_binary_1000-31frequencies_10-1layers_directdistance",
#     ),
# )
# params.append(
#     dict(
#         cls=chd.models.pred.model.better.Model,
#         n_frequencies=(1000, 500, 250, 125, 63, 31),
#         n_embedding_dimensions=100,
#         n_layers_fragment_embedder=5,
#         residual_fragment_embedder=False,
#         n_layers_embedding2expression=2,
#         residual_embedding2expression=False,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="radial_binary",
#         distance_encoder="direct",
#         label="radial_binary_1000-31frequencies_5-2layers_directdistance",
#     ),
# )
# params.append(
#     dict(
#         cls=chd.models.pred.model.better.Model,
#         n_frequencies=(1000, 500, 250, 125, 63, 31),
#         n_embedding_dimensions=100,
#         n_layers_fragment_embedder=0,
#         residual_fragment_embedder=False,
#         n_layers_embedding2expression=0,
#         residual_embedding2expression=False,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="radial_binary",
#         distance_encoder="direct",
#         label="radial_binary_1000-31frequencies_0-0layers_directdistance",
#         lr=1e-3,
#     ),
# )
# params.append(
#     dict(
#         cls=chd.models.pred.model.better.Model,
#         n_frequencies=(1000, 500, 250, 125, 63, 31),
#         n_embedding_dimensions=100,
#         n_layers_fragment_embedder=2,
#         residual_fragment_embedder=False,
#         n_layers_embedding2expression=2,
#         residual_embedding2expression=False,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="radial_binary",
#         distance_encoder="direct",
#         label="radial_binary_1000-31frequencies_2-2layers_directdistance",
#     ),
# )
# params.append(
#     dict(
#         cls=chd.models.pred.model.better.Model,
#         n_frequencies=(1000, 500, 250, 125, 63, 31),
#         n_embedding_dimensions=100,
#         n_layers_fragment_embedder=5,
#         residual_fragment_embedder=False,
#         n_layers_embedding2expression=5,
#         residual_embedding2expression=False,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="radial_binary",
#         distance_encoder="direct",
#         encoder_kwargs=dict(parameterize_loc=True),
#         label="radial_binary_locparam_1000-31frequencies_directdistance",
#     ),
# )
# params.append(
#     dict(
#         cls=chd.models.pred.model.better.Model,
#         n_frequencies=(1000, 500, 250, 125, 63, 31),
#         n_embedding_dimensions=100,
#         n_layers_fragment_embedder=5,
#         residual_fragment_embedder=False,
#         n_layers_embedding2expression=5,
#         residual_embedding2expression=False,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="radial_binary",
#         distance_encoder="split",
#         encoder_kwargs=dict(parameterize_loc=True),
#         label="radial_binary_locparam_1000-31frequencies_splitdistance",
#     ),
# )
# params.append(
#     dict(
#         cls=chd.models.pred.model.better.Model,
#         n_frequencies=5000,
#         n_embedding_dimensions=100,
#         n_layers_fragment_embedder=5,
#         residual_fragment_embedder=False,
#         n_layers_embedding2expression=5,
#         residual_embedding2expression=False,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="radial",
#         label="radial_5000frequencies",
#     ),
# )
# params.append(
#     dict(
#         cls=chd.models.pred.model.better.Model,
#         n_frequencies=1000,
#         n_embedding_dimensions=100,
#         n_layers_fragment_embedder=5,
#         residual_fragment_embedder=False,
#         n_layers_embedding2expression=5,
#         residual_embedding2expression=False,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="tophat",
#         label="tophat_1000frequencies",
#     ),
# )
# params.append(
#     dict(
#         cls=chd.models.pred.model.better.Model,
#         n_frequencies=5000,
#         n_embedding_dimensions=100,
#         n_layers_fragment_embedder=5,
#         residual_fragment_embedder=False,
#         n_layers_embedding2expression=5,
#         residual_embedding2expression=False,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="tophat",
#         label="tophat_5000frequencies",
#     ),
# )
# params.append(
#     dict(
#         cls=chd.models.pred.model.better.Model,
#         n_frequencies=10000,
#         n_embedding_dimensions=100,
#         n_layers_fragment_embedder=5,
#         residual_fragment_embedder=False,
#         n_layers_embedding2expression=5,
#         residual_embedding2expression=False,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="tophat",
#         label="tophat_10000frequencies",
#     ),
# )
# params.append(
#     dict(
#         cls=chd.models.pred.model.better.Model,
#         n_frequencies=100,
#         n_embedding_dimensions=100,
#         n_layers_fragment_embedder=5,
#         residual_fragment_embedder=False,
#         n_layers_embedding2expression=5,
#         residual_embedding2expression=False,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="radial",
#         n_cells_step=200,
#         label="radial_200cellsteps",
#     ),
# )
# params.append(
#     dict(
#         cls=chd.models.pred.model.better.Model,
#         n_frequencies=100,
#         n_embedding_dimensions=100,
#         n_layers_fragment_embedder=5,
#         residual_fragment_embedder=False,
#         n_layers_embedding2expression=5,
#         residual_embedding2expression=False,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="radial_binary",
#         n_cells_step=200,
#         label="radial_binary_200cellsteps",
#     ),
# )
# params.append(
#     dict(
#         cls=chd.models.pred.model.better.Model,
#         n_embedding_dimensions=5,
#         n_layers_fragment_embedder=1,
#         residual_fragment_embedder=False,
#         n_layers_embedding2expression=2,
#         residual_embedding2expression=False,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="exponential",
#         label="exponential",
#         lr=1e-2,
#     ),
# )

hashes = [hashlib.md5(str({k: v2 for k, v2 in v.items() if k not in ["label"]}).encode()).hexdigest() for v in params]
assert len(hashes) == len(set(hashes))

params = {h: v for h, v in zip(hashes, params)}
