# %%
import pathlib
import pandas as pd
import scanpy as sc
import scvelo as scv
import chromatinhd as chd
import chromatinhd.data

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('ticks')
scv.settings.verbosity = 3
scv.settings.presenter_view = True
scv.set_figure_params('scvelo')
sc._settings.ScanpyConfig.figdir = pathlib.PosixPath('')

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"
dataset_name = "hspc"
folder_data_preproc = folder_data / dataset_name

# %%
transcriptome = chromatinhd.data.Transcriptome(folder_data_preproc / "transcriptome")

#%%
df = pd.read_csv(folder_data_preproc / "GSM6403411_3423-MV-2_atac_fragments.tsv.gz", header=None, skiprows=51, nrows=200000000, sep="\t", compression="gzip")

#%%
df.columns = ['chrom', 'start', 'end', 'cell', 'count']

#%%
df_phase = transcriptome.adata.obs['phase']
df_phase.index.name = "cell"
df_phase = df_phase.reset_index()

#%%
df = pd.merge(df, df_phase, left_on='cell', right_on='cell', how='left')

#%% 
df['size'] = df['end'] - df['start']

#%%
grouped = df.groupby('phase')

#%%
for name, group in grouped:
    print(name, group.size)
    fig, ax = plt.subplots()
    ax.hist(group['size'], range = (0, 1000), bins = 100)
    ax.set_xlim(0, 1000)
    ax.set_title(name + ' phase')
    fig.savefig(f"{folder_data_preproc}/plots/{name}_phase.pdf")

#%%