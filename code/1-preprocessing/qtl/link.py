# %%
import chromatinhd as chd
import polyptich as pp
pp.setup_ipython()

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("ticks")

import pathlib
import pickle

import tqdm.auto as tqdm
# from chromatinhd_manuscript.designs_qtl import design
design = pd.DataFrame([
    ["lung", "100k100k", "hs/gwas", "gwas_asthma", "gwas_asthma", "GRCm39"]
], columns=["dataset", "regions", "folder_qtl", "qtl_name", "motifscan", "genome"])

# design = design.query("dataset == 'liver'")
print(design)

design = design.copy()
design["force"] = True

# %%

import chromatinhd.data.associations

for _, setting in design.iterrows():
    dataset_name = setting["dataset"]
    regions_name = setting["regions"]
    motifscan_name = setting["motifscan"]
    force = setting["force"]
    motifscan = chd.data.associations.Associations(
        chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name, reset=force
    )
    if not motifscan.scanned:
        force = True

    if force:
        print(setting)
        dataset_folder = chd.get_output() / "datasets" / dataset_name
        regions = chd.data.Regions(dataset_folder / "regions" / regions_name)
        chromosomes = regions.coordinates["chrom"].unique()

        folder_qtl = setting["folder_qtl"]
        qtl_name = setting["qtl_name"]

        snp_info = pickle.load((chd.get_output() / "snp_info.pkl").open("rb"))
        qtl_mapped = pd.read_pickle(
            chd.get_output() / "data" / "qtl" / folder_qtl / ("qtl_mapped_" + qtl_name + ".pkl")
        )
        qtl_mapped.index = np.arange(len(qtl_mapped))

        if "gtex" in qtl_name:
            association = qtl_mapped
            association["disease/trait"] = association["tissue"]
        else:
            association = qtl_mapped.join(snp_info, on="snp")
            print(association["start"])
            association = association.loc[~pd.isnull(association["start"])]
            association["pos"] = association["start"].astype(int)

        # filter on only main snp if necessary
        if motifscan_name.endswith("main"):
            print("filter")
            association = association.loc[association["snp_main"] == association["snp"]]

        print(association.query("snp == 'rs9391997'"))

        # liftover if necessary
        if setting["genome"] == "mm10":
            import liftover

            if "converter" not in globals():
                chain_file = chd.get_output() / "data" / "hg38ToMm10.over.chain.gz"
                if not chain_file.exists():
                    import os

                    os.system(
                        f"wget http://hgdownload.soe.ucsc.edu/goldenPath/hg38/liftOver/hg38ToMm10.over.chain.gz -O {chain_file}"
                    )
                converter = liftover.ChainFile(str(chain_file), "hg38", "mm10")
            association_old = association
            association_new = []
            for _, row in tqdm.tqdm(association.iterrows(), total=len(association)):
                converted = converter.convert_coordinate(row["chr"], row["pos"])
                if len(converted) == 1:
                    converted = converted[0]
                    row["chr"] = converted[0]
                    row["pos"] = converted[1]
                    association_new.append(row)
            association_new = pd.DataFrame(association_new)
            association = association_new
        elif setting["genome"] == "GRCm39":
            import liftover

            if "converter" not in globals():
                chain_file = chd.get_output() / "data" / "hg38ToMm39.over.chain.gz"
                if not chain_file.exists():
                    import os

                    os.system(
                        f"wget http://hgdownload.soe.ucsc.edu/goldenPath/hg38/liftOver/hg38ToMm39.over.chain.gz -O {chain_file}"
                    )
                converter = liftover.ChainFile(str(chain_file), "hg38", "Mm39")
            association_old = association
            association_new = []
            for _, row in tqdm.tqdm(association.iterrows(), total=len(association)):
                converted = converter.convert_coordinate(row["chr"], row["pos"])
                if row["snp"] in ['rs9391997', 'rs77315098', 'rs75763833']:
                    print(converted)
                if len(converted) == 1:
                    converted = converted[0]
                    row["chr"] = converted[0]
                    row["pos"] = converted[1]
                    association_new.append(row)
            association_new = pd.DataFrame(association_new)
            association = association_new

        # map to chromosomes
        import pybedtools

        association_bed = pybedtools.BedTool.from_dataframe(association.reset_index()[["chr", "pos", "pos", "index"]])

        coordinates = regions.coordinates
        coordinates["start"] = np.clip(coordinates["start"], 0, np.inf)
        coordinates_bed = pybedtools.BedTool.from_dataframe(coordinates[["chrom", "start", "end"]])
        intersection = association_bed.intersect(coordinates_bed)
        association = association.loc[intersection.to_dataframe()["name"].unique()]

        chromosome_mapping = pd.Series(np.arange(len(chromosomes)), chromosomes)
        coordinates["chr_int"] = chromosome_mapping[coordinates["chrom"]].values

        association = association.loc[association.chr.isin(chromosomes)].copy()
        association["chr_int"] = chromosome_mapping[association["chr"]].values
        association = association.sort_values(["chr_int", "pos"])

        assert np.all(np.diff(association["chr_int"].to_numpy()) >= 0), "Should be sorted by chr"

        motif_col = "disease/trait"
        association[motif_col] = association[motif_col].astype("category")

        # if differential, we only select eQTLs that affect differentially expressed genes
        if "differential" in motifscan_name:
            transcriptome = chd.data.Transcriptome(dataset_folder / "transcriptome")
            transcriptome = chd.data.Transcriptome(dataset_folder / "transcriptome")
            gene_ids = (
                transcriptome.var.sort_values("dispersions_norm", ascending=False).query("dispersions_norm > 1").index
            )

            if setting["genome"] == "mm10":
                # convert grch38 to mm10 gene ids
                association["gene"] = chd.biomart.get_orthologs(
                    chd.biomart.Dataset.from_genome("GRCh38"), association["gene"]
                )

            if "gene" not in association.columns:
                raise ValueError("No gene column found in association", association.columns)

            association = association.loc[association["gene"].isin(gene_ids)].copy()

        # do the actual mapping
        n = []

        coordinate_ixs = []
        region_ixs = []
        position_ixs = []
        motif_ixs = []
        scores = []

        for gene_ix, region_info in enumerate(coordinates.itertuples()):
            chr_int = region_info.chr_int
            chr_start = np.searchsorted(association["chr_int"].to_numpy(), chr_int)
            chr_end = np.searchsorted(association["chr_int"].to_numpy(), chr_int + 1)

            pos_start = chr_start + np.searchsorted(
                association["pos"].iloc[chr_start:chr_end].to_numpy(), region_info.start
            )
            pos_end = chr_start + np.searchsorted(
                association["pos"].iloc[chr_start:chr_end].to_numpy(), region_info.end
            )

            qtls_promoter = association.iloc[pos_start:pos_end].copy()
            qtls_promoter["relpos"] = qtls_promoter["pos"] - region_info.tss

            if region_info.strand == -1:
                qtls_promoter = qtls_promoter.iloc[::-1].copy()
                qtls_promoter["relpos"] = -qtls_promoter["relpos"]  # + regions.width + 1

            # if "rs10065637" in qtls_promoter["snp"].values.tolist():
            # print(gene_ix)
            # raise ValueError

            n.append(len(qtls_promoter))

            coordinate_ixs += (qtls_promoter["relpos"]).astype(int).tolist()
            region_ixs += [gene_ix] * len(qtls_promoter)
            position_ixs += (qtls_promoter["relpos"] + (gene_ix * regions.width)).astype(int).tolist()
            motif_ixs += (qtls_promoter[motif_col].cat.codes.values).astype(int).tolist()
            scores += [1] * len(qtls_promoter)

        motifs_oi = association[[motif_col]].groupby([motif_col]).first()
        motifs_oi["n"] = association.groupby(motif_col).size()

        # save
        motifscan.regions = regions
        motifscan.coordinates = np.array(coordinate_ixs)
        motifscan.region_indices = np.array(region_ixs)
        motifscan.indices = np.array(motif_ixs)
        motifscan.scores = np.array(scores)
        motifscan.strands = np.array([1] * len(motifscan.indices))
        motifscan.motifs = motifs_oi

        motifscan.create_region_indptr(overwrite=True)

        # association["snp_main"] = association["snp"]
        association["rsid"] = association["snp"]
        motifscan.association = association

        assert motifscan.scanned

# %%
association["snp_main"].value_counts()
# %%
