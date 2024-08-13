# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from IPython import get_ipython

if get_ipython():
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

import seaborn as sns

import tqdm.auto as tqdm

# %%
import chromatinhd as chd

# %% [markdown]
# For this, you have to go to the website, log in, try to download, get the link, and paste it below
# https://data.4dnucleome.org/files-processed/4DNFIXP4QG5B/

# %%
!wget https://4dn-open-data-public.s3.amazonaws.com/fourfront-webprod/wfoutput/d6abea45-b0bb-4154-9854-1d3075b98097/4DNFIXP4QG5B.mcool -O ~/NAS2/wsaelens/projects/chromatinhd/chromatinhd_manuscript/output/HiC/4DNFIXP4QG5B.mcool

# %%
# softlink
!ln -s ~/NAS2/wsaelens/projects/chromatinhd/chromatinhd_manuscript/output/HiC/4DNFIXP4QG5B.mcool {chd.get_output()}/HiC/4DNFIXP4QG5B.mcool

