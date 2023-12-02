params = []

import chromatinhd as chd
import chromatinhd.models.pred.model.better
import hashlib

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
        distance_encoder="direct",
        label="radial_binary_1000-31frequencies_directdistance",
    ),
)

params = {
    hashlib.md5(str({k: v2 for k, v2 in v.items() if k not in ["label"]}).encode()).hexdigest(): v for v in params
}
