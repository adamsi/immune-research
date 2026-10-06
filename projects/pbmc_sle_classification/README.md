# 🧬 PBMC SLE Classification (planned)

Classify systemic lupus erythematosus (SLE) patients from healthy donors using peripheral blood
single-cell RNA-seq, at the patient level, and explain which cell states drive the call.

## Plan

1. **Data**: Perez et al. 2022 (*Science*) lupus PBMC atlas, `.h5ad` from CELLxGENE.
2. **Preprocessing**: QC filtering, normalisation, highly variable genes; batch-effect stratification
   (batch/cohort and processing date, per condition) and removal with scVI `batch_key` (or Harmony).
3. **Representation**: scVI latent space, CellTypist cell types, UMAP atlas.
4. **Models**: pseudobulk logistic-regression baseline vs attention-MIL (`shared/mil_torch.py`),
   donor-split cross-validation, AUROC.
5. **Insights**: attention per cell type (`shared/plotting.py`) and pseudobulk differential expression.

Raw data and `.h5ad` files stay out of git (see the root `.gitignore`).
