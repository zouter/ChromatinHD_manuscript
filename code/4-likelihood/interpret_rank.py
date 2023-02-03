import pandas as pd
import numpy as np

import chromatinhd as chd
import chromatinhd.scorer

import pickle

import torch
import tqdm.auto as tqdm

device = "cuda:0"

folder_root = chd.get_output()
folder_data = folder_root / "data"

for dataset_name in [
    "pbmc10k",
    "lymphoma",
    "e18brain",
    "alzheimer",
    "brain",
]:
    print(f"{dataset_name=}")
    # transcriptome
    folder_data_preproc = folder_data / dataset_name

    # fragments
    # promoter_name, window = "1k1k", np.array([-1000, 1000])
    promoter_name, window = "10k10k", np.array([-10000, 10000])
    # promoter_name, window = "20kpromoter", np.array([-10000, 0])
    promoters = pd.read_csv(
        folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
    )
    window_width = window[1] - window[0]

    fragments = chd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)
    fragments.window = window

    # create design to run
    from design import get_design, get_folds_inference

    class Prediction(chd.flow.Flow):
        pass

    # folds & minibatching
    folds = pickle.load((fragments.path / "folds.pkl").open("rb"))
    folds = get_folds_inference(fragments, folds)

    for latent_name in [
        # "leiden_1"
        "leiden_0.1"
    ]:
        latent_folder = folder_data_preproc / "latent"
        latent = torch.from_numpy(
            pickle.load((latent_folder / (latent_name + ".pkl")).open("rb")).values
        ).to(torch.float)

        # create design to run
        from design import get_design, get_folds_inference

        design = get_design(dataset_name, latent_name, fragments)
        design = {
            k: design[k]
            for k in [
                # "v4_128-64-32",
                # "v4_256-128-64-32",
                "v4_128",
                "v4_64",
                "v4_32",
                "v4_64-32",
            ]
        }
        fold_slice = slice(0, 1)

        # folds & minibatching
        folds = pickle.load((fragments.path / "folds.pkl").open("rb"))
        folds = get_folds_inference(fragments, folds)

        for prediction_name, design_row in design.items():
            print(f"{dataset_name=} {promoter_name=} {prediction_name=}")
            prediction = Prediction(
                chd.get_output()
                / "prediction_likelihood"
                / dataset_name
                / promoter_name
                / latent_name
                / prediction_name
            )

            ## Single base-pair inference
            # load all models
            models = [
                pickle.load(
                    open(prediction.path / ("model_" + str(fold_ix) + ".pkl"), "rb")
                )
                for fold_ix, fold in enumerate(folds[fold_slice])
            ]
            model = models[0]

            # device = "cuda"
            # model = model.to(device).eval()

            # # create design for inference
            # design_gene = pd.DataFrame({"gene_ix":np.arange(fragments.n_genes)})
            # design_latent = pd.DataFrame({"active_latent":np.arange(latent.shape[1])})
            # design_coord = pd.DataFrame({"coord":np.arange(window[0], window[1]+1, step = 25)})
            # design = chd.utils.crossing(design_gene, design_latent, design_coord)
            # design["batch"] = np.floor(np.arange(design.shape[0]) / 10000).astype(int)

            # # infer
            # probs = []
            # rho_deltas = []
            # rhos = []
            # for _, design_subset in tqdm.tqdm(design.groupby("batch")):
            #     pseudocoordinates = torch.from_numpy(design_subset["coord"].values).to(device)
            #     pseudocoordinates = (pseudocoordinates - window[0]) / (window[1] - window[0])
            #     pseudolatent = torch.nn.functional.one_hot(torch.from_numpy(design_subset["active_latent"].values).to(device), latent.shape[1]).to(torch.float)
            #     gene_ix = torch.from_numpy(design_subset["gene_ix"].values).to(device)

            #     likelihood_mixture, rho_delta, rho = model.evaluate_pseudo(pseudocoordinates.to(device), latent = pseudolatent.to(device), gene_ix = gene_ix)
            #     prob_mixture = likelihood_mixture.detach().cpu().numpy()
            #     rho_delta = rho_delta.detach().cpu().numpy()
            #     rho = rho.detach().cpu().numpy()

            #     probs.append(prob_mixture)
            #     rho_deltas.append(rho_delta)
            #     rhos.append(rho)
            # probs = np.hstack(probs)
            # rho_deltas = np.hstack(rho_deltas)
            # rhos = np.hstack(rhos)

            # probs = probs.reshape((design_gene.shape[0], design_latent.shape[0], design_coord.shape[0]))
            # rho_deltas = rho_deltas.reshape((design_gene.shape[0], design_latent.shape[0], design_coord.shape[0]))
            # rhos = rhos.reshape((design_gene.shape[0], design_latent.shape[0], design_coord.shape[0]))

            # # calculate the score we're gonna use: how much does the likelihood of a cut in a window change compared to the "mean"?
            # probs_diff = probs - probs.mean(-2, keepdims = True) + rho_deltas# - rho_deltas.mean(-2, keepdims = True)

            # # apply a mask to regions with very low likelihood of a cut
            # rho_cutoff = np.log(1.)
            # mask = rhos > rho_cutoff

            # probs_diff_masked = probs_diff.copy()
            # probs_diff_masked[~mask] = -np.inf

            # import gzip

            # pickle.dump({
            #     "probs_diff":probs_diff,
            #     "rhos":rhos,
            #     "coord":(design["coord"].values)
            # }, gzip.GzipFile((prediction.path / "basepair_ranking.pkl"), "wb"))

            # ## Single base-pair resolution
            # # interpolate the scoring from above but now at single base pairs
            # # we may have to smooth this in the future, particularly for very detailed models that already look at base pair resolution
            # x = (design["coord"].values).reshape((design_gene.shape[0], design_latent.shape[0], design_coord.shape[0]))
            # y = probs_diff

            # def interpolate(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
            #     a = (fp[...,1:] - fp[...,:-1]) / (xp[...,1:] - xp[...,:-1])
            #     b = fp[..., :-1] - (a.mul(xp[..., :-1]) )

            #     indices = torch.searchsorted(xp.contiguous(), x.contiguous(), right=False) - 1
            #     indices = torch.clamp(indices, 0, a.shape[-1] - 1)
            #     slope = a.index_select(a.ndim-1, indices)
            #     intercept = b.index_select(a.ndim-1, indices)
            #     return x * slope + intercept

            # desired_x = torch.arange(*window)

            # probs_diff_interpolated = interpolate(desired_x, torch.from_numpy(x)[0][0], torch.from_numpy(probs_diff)).numpy()
            # rhos_interpolated = interpolate(desired_x, torch.from_numpy(x)[0][0], torch.from_numpy(rhos)).numpy()

            # # again apply a mask
            # rho_cutoff = np.log(1.)
            # probs_diff_interpolated_masked = probs_diff_interpolated.copy()
            # mask_interpolated = (rhos_interpolated > rho_cutoff)
            # probs_diff_interpolated_masked[~mask_interpolated] = -np.inf

            # # store this scoring
            # pickle.dump(probs_diff_interpolated_masked, (prediction.path / "basepair_ranking.pkl").open("wb"))

            probs_diff_interpolated_masked = model.rank(window, latent.shape[1])

            # # score
            # scorer = chd.scoring.likelihood.Scorer(models, folds[:len(models)], loaders = loaders, device = device, gene_ids = fragments.var.index, cell_ids = fragments.obs.index)
            # scores, genescores = scorer.score()
