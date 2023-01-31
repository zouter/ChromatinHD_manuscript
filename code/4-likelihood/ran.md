# PBMC10k

PReprocessing:
- Main preproc ok
- Celltype latent ok (leiden 0.1)
- macs2 ok
- mac2_improved ok
- genrich ok
- macs2_improved_leiden_0.1 ok

- peak counts nok (genrich)
- main model ok


# E18brain

PReprocessing:
- Main preproc ok
- Celltype latent nok
- Leiden 0.1 ok
- macs2 ok
- mac2_improved ok
- genrich ok still need to transfer
- macs2_improved_leiden_0.1 nok (busy)

- peak counts nok (genrich, macs2_improved)
- main model ok

# Lymphoma

PReprocessing:
- Main preproc ok
- Celltype latent ok
- macs2 ok
- mac2_improved ok
- genrich ok still need to transfer
- macs2_improved_celltype ok

- motifscan cutoff_001 nok

- peak counts nok (macs2_improved)
- main model ok

# Alzheimer
Preprocessing
- Main preproc ok
- Leiden 0.1 ok
- Celltype latent nok
- macs2 ok
- mac2_improved ok
- genrich nok, impossible to run as no bam is available
- macs2_improved_leiden_0.1 nok

# Brain
Preprocessing
- Main preproc nok
- Leiden 0.1 ok
- Celltype latent nok
- macs2 ok
- macs2_improved ok
- genrich ok
- macs2_improved_leiden_0.1 ok

Interpret motif enrichment has to be rerun for everything