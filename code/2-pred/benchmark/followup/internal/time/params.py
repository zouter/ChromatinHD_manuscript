params = []

import chromatinhd as chd
import chromatinhd.models.pret.model.better
import chromatinhd.models.pred.model.better
import hashlib

for dt in [0, 0.05, 0.1, 0.2]:
    params.append(
        dict(
            cls=chd.models.pret.model.better.Model,
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
            lr=1e-4,
            delta_time=dt,
            label=f"dt{dt}/radial_binary_1000-31frequencies_splitdistance",
        ),
    )
    params.append(
        dict(
            cls=chd.models.pret.model.better.Model,
            n_frequencies=(1000, 500, 250, 125, 63, 31),
            n_embedding_dimensions=100,
            n_layers_fragment_embedder=5,
            residual_fragment_embedder=False,
            n_layers_embedding2expression=5,
            residual_embedding2expression=False,
            dropout_rate_fragment_embedder=0.0,
            dropout_rate_embedding2expression=0.0,
            encoder="radial_binary",
            lr=1e-4,
            delta_time=dt,
            label=f"dt{dt}/radial_binary_1000-31frequencies",
        ),
    )
    params.append(
        dict(
            cls=chd.models.pret.model.better.Model,
            n_embedding_dimensions=5,
            n_layers_fragment_embedder=1,
            residual_fragment_embedder=False,
            n_layers_embedding2expression=2,
            residual_embedding2expression=False,
            dropout_rate_fragment_embedder=0.0,
            dropout_rate_embedding2expression=0.0,
            encoder="exponential",
            lr=1e-3,
            delta_time=dt,
            label=f"dt{dt}/exponential",
        ),
    )


for dt in [0.05]:
    params.append(
        dict(
            cls=chd.models.pret.model.better.Model,
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
            lr=1e-3,
            delta_time=dt,
            delta_expression=True,
            label=f"dxdt{dt}/radial_binary_1000-31frequencies_splitdistance",
        ),
    )
    params.append(
        dict(
            cls=chd.models.pret.model.better.Model,
            n_embedding_dimensions=5,
            n_layers_fragment_embedder=1,
            residual_fragment_embedder=False,
            n_layers_embedding2expression=2,
            residual_embedding2expression=False,
            dropout_rate_fragment_embedder=0.0,
            dropout_rate_embedding2expression=0.0,
            encoder="exponential",
            lr=1e-3,
            delta_time=dt,
            delta_expression=True,
            label=f"dxdt{dt}/exponential",
        ),
    )

params.append(
    dict(
        cls=chd.models.pred.model.better.Model,
        n_embedding_dimensions=5,
        n_layers_fragment_embedder=1,
        residual_fragment_embedder=False,
        n_layers_embedding2expression=2,
        residual_embedding2expression=False,
        dropout_rate_fragment_embedder=0.0,
        dropout_rate_embedding2expression=0.0,
        encoder="exponential",
        label="pred/exponential",
        lr=1e-2,
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
        label="pred/radial_binary_1000-31frequencies_splitdistance",
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
#         label="test",
#     ),
# )


params = {
    hashlib.md5(str({k: v2 for k, v2 in v.items() if k not in ["label"]}).encode()).hexdigest(): v for v in params
}
