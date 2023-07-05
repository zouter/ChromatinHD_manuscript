import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

default_figsize = (15, 15)
default_dpi = 100
default_titlesize = 20
default_labelsize = 18

def cutsites(gene, df, directory, show=False):
    fig, ax = plt.subplots(figsize=default_figsize)
    ax.scatter(df['x'], df['y'], s=1, marker='s', color='black')
    ax.set_title(f"{gene} (cut sites = {len(df)})", fontsize=default_titlesize) #check if len(df) is correct
    ax.set_xlabel('Position', fontsize=default_labelsize)
    ax.set_ylabel('Latent Time', fontsize=default_labelsize)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_facecolor('white')

    os.makedirs(directory, exist_ok=True)
    plt.savefig(directory / f'{gene}.png', dpi=default_dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)

def cutsites_histo(gene, df, df_long, n_fragments, directory, show=False):
    fig, axs = plt.subplots(figsize=default_figsize, ncols=2, gridspec_kw={'width_ratios': [1, 3]})

    ax_hist = axs[0]
    ax_hist.hist(df['rank'], bins=100, orientation='horizontal')
    ax_hist.set_xlabel('n cells')
    ax_hist.set_ylabel('Rank')
    ax_hist.set_ylim([0, 1])
    ax_hist.invert_xaxis()

    ax_scatter = axs[1]
    ax_scatter.scatter(df_long['x'], df_long['y'], s=1, marker='s', color='black')
    ax_scatter.set_xlabel('Position')
    ax_scatter.set_ylabel('Latent Time')
    ax_scatter.set_xlim([0, 1])
    ax_scatter.set_ylim([0, 1])
    ax_scatter.set_facecolor('white')

    fig.suptitle(f"{gene} (cut sites = {2 * n_fragments})", fontsize=20)

    os.makedirs(directory, exist_ok=True)
    plt.savefig(directory / f'{gene}.png', dpi=default_dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)

def model_continuous(gene, directory, show=False):
    file_name = directory / f"{gene}.csv"
    probsx = np.loadtxt(file_name, delimiter=',')

    dir_plot_full = directory.parent / 'plots' / directory.name
    file_name = dir_plot_full / f"{gene}.png"

    fig, ax = plt.subplots(figsize=default_figsize)
    heatmap = ax.imshow(probsx, cmap='RdBu_r', aspect='auto')
    cbar = plt.colorbar(heatmap)
    ax.set_title(f'Probs for gene_oi = {gene}', fontsize=default_titlesize)
    ax.set_xlabel('Position', fontsize=default_labelsize)
    ax.set_ylabel('Latent Time', fontsize=default_labelsize)

    os.makedirs(dir_plot_full, exist_ok=True)
    plt.savefig(file_name, dpi=default_dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)

def minmax(gene, df, directory, show=False):
    minima = df[df['type'] == 'minima']
    maxima = df[df['type'] == 'maxima']
    
    plt.figure(figsize=default_figsize)
    plt.scatter(minima['x'], minima['y'], c='red', s=1, linewidths=0.5, label='Minima')
    plt.scatter(maxima['x'], maxima['y'], c='blue', s=1, linewidths=0.5, label='Maxima')
    plt.title(str(directory).split('/')[-1], fontsize=default_titlesize)
    plt.xlabel('Positions', fontsize=default_labelsize)
    plt.ylabel('Latent Time', fontsize=default_labelsize)
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    os.makedirs(directory, exist_ok=True)
    plt.savefig(directory / f"{gene}.png", dpi=default_dpi)

    if show:
        plt.show()
    else:
        plt.close()

def model_quantile(gene_oi, latent_torch, fragments, directory, show=False):
    gene_id = fragments.var.index[gene_oi]

    file_name = directory / f"{gene_id}.csv"
    probs = pd.read_csv(file_name)

    dir_plot_full = directory.parent / 'plots' / directory.name
    file_name = dir_plot_full / f"{gene_id}.png"

    bins = np.linspace(0, 1, 500)
    binmids = (bins[1:] + bins[:-1])/2
    binsize = binmids[1] - binmids[0]
    pseudocoordinates = torch.linspace(0, 1, 1000)

    fig, axes = plt.subplots(probs.shape[0], 1, figsize=(20, 1*probs.shape[0]), sharex = True, sharey = True)
    for i, ax in zip(reversed(range(probs.shape[0])), axes):
        n_cells = latent_torch[:, i].sum()

        fragments_oi = (latent_torch[fragments.cut_local_cell_ix, i] != 0) & (fragments.cut_local_gene_ix == gene_oi)
        bincounts, _ = np.histogram(fragments.cut_coordinates[fragments_oi].cpu().numpy(), bins = bins)
        freq = round((fragments_oi.sum()/n_cells).item(), 3)

        ax.bar(binmids, bincounts / n_cells * len(bins), width = binsize, color = "#888888", lw = 0)
        ax.plot(pseudocoordinates.numpy(), probs.iloc[i, 1:], label = i, color = "#0000FF", lw = 2, zorder = 20)
        ax.plot(pseudocoordinates.numpy(), probs.iloc[i, 1:], label = i, color = "#FFFFFF", lw = 3, zorder = 10)
        ax.set_ylabel(f"{probs.iloc[i]['cluster']}\n freq={freq}", rotation = 0, ha = "right", va = "center")

    fig.suptitle(f'Probs for gene_oi = {gene_id}', y=0.92, fontsize=default_titlesize)
    fig.text(0.5, 0.03, "Position", ha="center", fontsize=default_labelsize)
    fig.text(0.04, 0.5, "Likelihood", va="center", rotation="vertical", fontsize=default_labelsize)

    os.makedirs(dir_plot_full, exist_ok=True)
    plt.savefig(file_name)

    if show:
        plt.show()
    else:
        plt.close(fig)

def combine_1rows_3cols(dirs, output_dir):

    files = []
    for directory in dirs:
        files_in_dir = sorted([f for f in os.listdir(directory) if f.endswith(".png")])
        files.append(files_in_dir)
    
    # check if all files have the same length
    if not all(len(file) == len(files[0]) for file in files):
        raise ValueError("All files must have the same length.")

    # Determine the number of rows and columns
    nrows = 1
    ncols = 3

    for i, (file1, file2, file3) in enumerate(zip(*files)):
        # Open images from each directory
        img1 = Image.open(os.path.join(dirs[0], file1))
        img2 = Image.open(os.path.join(dirs[1], file2))
        img3 = Image.open(os.path.join(dirs[2], file3))

        # Create a blank canvas for the combined image grid
        grid_width = max(img1.width, img2.width, img3.width)
        grid_height = max(img1.height, img2.height, img3.height)
        combined_img = Image.new("RGB", (grid_width * ncols, grid_height * nrows))

        # Paste images into the combined image grid
        combined_img.paste(img1, (grid_width * 0, grid_height * 0))
        combined_img.paste(img2, (grid_width * 1, grid_height * 0))
        combined_img.paste(img3, (grid_width * 2, grid_height * 0))

        # Save the combined image
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, file1)
        combined_img.save(output_file, "PNG")

        print(output_file)

