params = []

import chromatinhd as chd
import chromatinhd.models.pred.model.shared
import hashlib

# params.append(
#     dict(
#         cls=chd.models.pred.model.shared.Model,
#         n_frequencies=20,
#         n_embedding_dimensions=19,
#         n_layers_fragment_embedder=1,
#         n_layers_embedding2expression=1,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="radial",
#         distance_encoder=None,
#         label="radial_distancenone",
#     ),
# )

# params.append(
#     dict(
#         cls=chd.models.pred.model.shared.Model,
#         n_frequencies=100,
#         n_embedding_dimensions=100,
#         n_layers_fragment_embedder=1,
#         n_layers_embedding2expression=1,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="direct",
#         distance_encoder=None,
#         label="distancenone",
#     ),
# )

# params.append(
#     dict(
#         cls=chd.models.pred.model.shared.Model,
#         n_frequencies=100,
#         n_embedding_dimensions=100,
#         n_layers_fragment_embedder=1,
#         n_layers_embedding2expression=1,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="direct",
#         distance_encoder="direct",
#         label="distancedirect",
#     ),
# )


params.append(
    dict(
        cls=chd.models.pred.model.shared.Model,
        n_frequencies=100,
        n_embedding_dimensions=100,
        n_layers_fragment_embedder=1,
        n_layers_embedding2expression=1,
        dropout_rate_fragment_embedder=0.0,
        dropout_rate_embedding2expression=0.0,
        encoder=None,
        distance_encoder="direct",
        label="none_distancedirect",
    ),
)


# params.append(
#     dict(
#         cls=chd.models.pred.model.shared.Model,
#         n_frequencies=100,
#         n_embedding_dimensions=100,
#         n_layers_fragment_embedder=0,
#         n_layers_embedding2expression=1,
#         dropout_rate_fragment_embedder=0.0,
#         dropout_rate_embedding2expression=0.0,
#         encoder="ones",
#         label="counter",
#     ),
# )

params = {
    hashlib.md5(str({k: v2 for k, v2 in v.items() if k not in ["label"]}).encode()).hexdigest(): v for v in params
}
