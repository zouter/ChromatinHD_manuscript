# %%
import torch

# %%
x = torch.arange(3 * 877).reshape(3, 877)

# %%
localcells = torch.tensor([0, 1])
genes = torch.tensor([3, 3])
# %%
localcellxgene = torch.tensor([3, 6])

x.flatten()[localcellxgene]
# %%
localcells = torch.tensor([0, 1])
genes = torch.tensor([3, 3])
# %%
n_local_cells = 3
localcellxgene2 = localcells * n_local_cells + genes
# %%
x[localcells, genes]
# %%
x.flatten()[localcellxgene]
# %%
