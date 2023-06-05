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

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")

import tqdm.auto as tqdm
import xarray as xr

# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

# %%
e1_e2_distance = 20
# e1_b1_distance, e2_b2_distance = -5, -5
e1_b1_distance, e2_b2_distance = 5, -5
e1_b1_distance, e2_b2_distance = -5, 5
e1_b1_distance, e2_b2_distance = 2, -15
e1_b1_distance, e2_b2_distance = 5, 5
e1_b1_distance, e2_b2_distance = -10, -22
e1_b1_distance, e2_b2_distance = 15, -5
e1_b1_distance, e2_b2_distance = 1, 50
e1_b1_distance, e2_b2_distance = -40, -2


# e1_b1_distance, e2_b2_distance = 15, 5
# e1_b1_distance, e2_b2_distance = -50, -1
# e1_b1_distance, e2_b2_distance = 1, 2
# e1_b1_distance, e2_b2_distance = 40, -5
# e1_b1_distance, e2_b2_distance = 50, -2

e1_position = -e1_e2_distance / 2
e2_position = e1_e2_distance / 2
b1_position = e1_position + e1_b1_distance
b2_position = e2_position + e2_b2_distance


# %%
fig, ax = plt.subplots()
circle = mpl.patches.Circle((e1_position, 0), 0.5, color="black")
ax.add_patch(circle)
circle = mpl.patches.Circle((e2_position, 0), 0.5, color="black")
ax.add_patch(circle)
ax.text(0, 1, e1_e2_distance, ha="center", va="center")

ax.set_aspect("equal")

circle = mpl.patches.Rectangle((b1_position, -0.25), 0.5, 0.5, color="red")
ax.add_patch(circle)
ax.text(0, 1, e1_e2_distance, ha="center", va="center")
circle = mpl.patches.Rectangle((b2_position, -0.25), 0.5, 0.5, color="red")
ax.add_patch(circle)
ax.text(0, 1, e1_e2_distance, ha="center", va="center")

ax.set_xlim(min(e1_position, b1_position) - 1, max(e2_position, b2_position) + 1)
ax.set_ylim(-1, 1)

# %%
