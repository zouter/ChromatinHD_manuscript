# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpyro
import jax
numpyro.set_platform("cuda")

# %%
# # !pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# %%
data = pd.read_table("data.txt", sep = " ")

# %%
from scipy import interpolate

def f(x):
    x_points = [ 0, 1, 2, 3, 4, 5]
    y_points = [12,14,22,39,58,77]

    tck = interpolate.splrep(x_points, y_points)
    print(tck)
    return interpolate.splev(x, tck)

print(f(1.25))

# %%
import numpyro
from numpyro.infer import MCMC, NUTS, Predictive
import numpyro.distributions as dist
from jax import random

assert numpyro.__version__.startswith("0.11.0")
def model(barcodes, gdna_barcode_counts = None):
    rate = numpyro.param("rate", 1, constraint = numpyro.distributions.constraints.positive)
    dispersion_gdna = numpyro.param("dispersion_gdna", 1, constraint = numpyro.distributions.constraints.positive)
    
    beta = numpyro.param("beta", 1, constraint = numpyro.distributions.constraints.positive)
    beta = numpyro.param("beta", 1, constraint = numpyro.distributions.constraints.positive)
    
    with numpyro.plate("barcodes", len(barcodes)):                
        mu_gdna = numpyro.sample("mu_gdna", dist.Exponential(rate))
        numpyro.sample("gdna", dist.NegativeBinomial2(mu_gdna, dispersion_gdna), obs=gdna_barcode_counts)
        
        

# %%
data_oi = data[["gDNA_raw1", "kmer"]].copy().dropna()

# %%
nuts_kernel = NUTS(model)

mcmc = MCMC(nuts_kernel, num_samples=100, num_warmup=500)
rng_key = random.PRNGKey(0)
mcmc.run(rng_key, data_oi["kmer"].values, gdna_barcode_counts = data_oi["gDNA_raw1"].values)

posterior_samples = mcmc.get_samples()

# %%
import arviz as az

# %%
data.posterior.mu_gdna.sel(chain = 0).to_pandas().mean(0)

# %%
data = az.from_numpyro(mcmc)
az.plot_trace(data, compact=True, figsize=(15, 45));
