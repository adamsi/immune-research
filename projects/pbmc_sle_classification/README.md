# 🧬 PBMC SLE Classification (planned)

Classify systemic lupus erythematosus (SLE) patients from healthy donors using peripheral blood
single-cell RNA-seq, at the patient level, and explain which cell states drive the call.

## Plan

1. **Data**: Perez et al. 2022 (*Science*) lupus PBMC atlas, downloaded as `.h5ad` from CELLxGENE.
2. **QC**: scanpy quality control (genes/counts/mito filters), normalisation, highly variable genes.
   **Batch effect stratification**: check and report batch/cohort and processing-date effects stratified by
   condition, then remove them (scVI with `batch_key`, or Harmony as an alternative).
3. **Latent space and cell types**: scVI embedding (unsupervised); cell-type labels from CellTypist (or Azimuth).
4. **Baseline**: pseudobulk per patient (and per cell type), then logistic regression.
5. **Attention-MIL**: each patient is a bag of cells, reusing `shared/mil_torch.py`.
6. **Evaluation**: patient-split cross-validation (no patient in both train and test), AUROC with CI.
7. **Insights**: attention share by cell type (`shared/plotting.py`) plus pseudobulk differential expression.

Raw data and `.h5ad` files stay out of git (see the root `.gitignore`).
