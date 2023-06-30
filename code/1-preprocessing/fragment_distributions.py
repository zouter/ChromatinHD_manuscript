# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Fragment distributions across datasets

# %%
import gzip

# %%
dataset_names = ["pbmc10k_gran", "pbmc10k", "pbmc3k", "lymphoma", "e18brain"]

# %%
sizes = {}
for dataset_name in dataset_names:
    sizes_dataset = []
    with gzip.GzipFile(folder_data / dataset_name / "atac_fragments.tsv.gz", "r") as fragment_file:
        i = 0
        for line in fragment_file:
            line = line.decode("utf-8")
            if line.startswith("#"):
                continue
            split = line.split("\t")
            sizes_dataset.append(int(split[2]) - int(split[1]))
            i += 1
            if i > 1000000:
                break
    sizes[dataset_name] = sizes_dataset

# %%
bins = np.linspace(0, 1000, 100+1)

# %%
bincounts = {dataset_name:np.histogram(x, bins, density = True)[0] for dataset_name, x in sizes.items()}


# %%
def ecdf(a):
    x = np.sort(a)
    y = np.arange(len(x))/float(len(x))
    return y


# %%
fig, ax = plt.subplots()
for dataset_name, bincounts_dataset in bincounts.items():
    x = np.sort(sizes[dataset_name])
    x_ecdf = ecdf(x)
    ax.plot(x, x_ecdf, label = dataset_name)
    ax.set_xscale("log")
ax.legend()
ax.set_xlabel("fragment length")
ax.set_ylabel("ECDF", rotation = 0, ha = "right", va = "center")

# %%
fig, ax = plt.subplots()
for dataset_name in sizes.keys():
    ax.hist(sizes[dataset_name], range = (0, 1000), bins = 100, histtype = "step", label = dataset_name)
    ax.set_xlim(0, 1000)
plt.legend()
ax.set_xlabel("fragment length")
ax.set_ylabel("# fragments", rotation = 0, ha = "right", va = "center")
