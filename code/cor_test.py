# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
n_vcms = 18000
n_samples = 72
x = np.random.random((n_vcms, n_samples))

# %%
cors = np.corrcoef(x)

# %%
plt.hist(cors.flatten(), bins=100)

# %%
np.fill_diagonal(cors, 0)

# %%
cors.max()

# %%
n_significant_vcms = ((cors > 0.5).sum(0) > 0).sum(0) / n_vcms
fig, ax = plt.subplots(figsize=(0.5, 4))
ax.bar(0, 1 - n_significant_vcms)
ax.bar(0, n_significant_vcms, bottom=1 - n_significant_vcms)

# %%
