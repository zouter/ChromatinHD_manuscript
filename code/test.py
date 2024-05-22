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

import torch

import pickle

import scanpy as sc

import tqdm.auto as tqdm
import io

import chromatinhd as chd

folder_root = chd.get_output()
folder_data = folder_root / "data"
dataset_name = "hspc"

folder_data_preproc = folder_data / dataset_name / "MV2"
folder_data_preproc.mkdir(exist_ok=True, parents=True)
genome = "GRCh38"

folder_dataset = chd.get_output() / "datasets" / "hspc"

import scanpy as sc

dataset_folder = chd.get_output() / "datasets" / "hspc"
transcriptome = chd.data.transcriptome.Transcriptome(path=dataset_folder / "transcriptome")

adata = pickle.load((folder_data_preproc / "adata.pkl").open("rb"))[transcriptome.adata.obs.index]
adata.obs = transcriptome.obs
adata.obsm["X_umap2"] = transcriptome.adata.obsm["X_umap"]


def gene_id(symbols):
    return adata.var.reset_index().set_index("symbol").loc[symbols, "gene"].values


plotdata = transcriptome.adata.obs.copy()
plotdata["expression"] = sc.get.obs_df(adata, gene_id(["SPI1"]), layer="normalized")

plotdata["umap1"] = np.array(adata.obsm["X_umap2"][:, 0])
plotdata["umap2"] = np.array(adata.obsm["X_umap2"][:, 1])
plotdata = plotdata.iloc[:1000]


import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd

# Example data
# df = pd.DataFrame({
#     'x': np.random.randint(0, 100, 100),
#     'y': np.random.randint(0, 100, 100),
#     'customdata': np.random.choice(['A', 'B', 'C'], 100),
# })

# Create a scatter plot
fig = go.Figure(
    data=go.Scatter(x=plotdata["umap1"], y=plotdata["umap2"], mode="markers", customdata=plotdata["expression"])
)
fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), width=800, height=800, plot_bgcolor="white")

# Initialize the Dash app
app = dash.Dash("test")

# Layout of the app
app.layout = html.Div(
    [
        dcc.Graph(id="scatter-plot", figure=fig),
        html.Pre(id="selected-data", style={"whiteSpace": "pre-line"}),
    ]
)


# Callback to update the selection
@app.callback(Output("selected-data", "children"), [Input("scatter-plot", "selectedData")])
def display_selected_data(selectedData):
    if selectedData is None:
        return "No data selected"
    print(selectedData["lassoPoints"])
    return "Hi"
    # return str(selectedData["lassoPoints"])
    # return str(selectedData)


app.run_server(debug=True)
