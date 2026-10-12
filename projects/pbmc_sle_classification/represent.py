"""Step 2: scVI latent space (batch removed), a CellTypist model of Perez's cell labels, UMAP atlas.

Training cells keep Perez's own labels (obs["cell_type_fine"]). A CellTypist model is trained on them so the
demo can give a new donor's cells the same labels.

Input : data/processed/pbmc.h5ad
Output: data/processed/pbmc_rep.h5ad   + obsm["X_scVI"], obsm["X_umap"]
        artifacts/genes.txt            the HVGs scVI reads, in order
        artifacts/scvi_encoder.npz     scVI encoder weights as numpy (the app embeds new patients with it)
        artifacts/celltypist.npz       CellTypist classifier of Perez labels as numpy (the app labels new cells with it)
        artifacts/scvi/                full scVI model (for reuse in notebooks)
        results/umap.png               atlas coloured by cell type, disease state and batch
"""
import os

import anndata as ad
import matplotlib
import numpy as np
import scanpy as sc

from shared.common import load_config, parse_args, set_seed

from . import DEFAULT_CONFIG, path
from .features import export_celltypist, export_scvi_encoder

matplotlib.use("Agg")


def train_scvi(adata: ad.AnnData, batch_key: str | None, r: dict):
    """scVI on raw counts of the HVGs. Batch and continuous covariates (e.g. % mito, which Perez regressed out)
    go to the decoder only (encode_covariates=False), so the encoder is covariate-free and can embed a new
    patient from an unseen batch with counts alone."""
    import scvi

    hvg = adata[:, adata.var["highly_variable"]].copy()
    cont = [c for c in r.get("continuous_covariates") or [] if c in hvg.obs]
    scvi.model.SCVI.setup_anndata(hvg, layer="counts", batch_key=batch_key, continuous_covariate_keys=cont or None)
    vae = scvi.model.SCVI(hvg, n_latent=r["n_latent"], n_hidden=r["n_hidden"], n_layers=r["n_layers"],
                          gene_likelihood=r["gene_likelihood"], encode_covariates=False)
    vae.train(max_epochs=r["scvi_epochs"])
    return vae, hvg.var_names.tolist()


def train_celltypist(adata: ad.AnnData, label_key: str, max_per_type: int, seed: int):
    """CellTypist (logistic regression on log-normalised expression, all genes) trained on Perez's labels,
    with up to max_per_type cells per type so rare types are not drowned out."""
    import celltypist

    rng = np.random.default_rng(seed)
    idx = np.sort(np.concatenate([rng.choice(ix, min(len(ix), max_per_type), replace=False)
                                  for ix in adata.obs.groupby(label_key, observed=True).indices.values()]))
    sub = adata[idx].copy()
    print(f"  CellTypist: {sub.n_obs:,} cells, {sub.obs[label_key].nunique()} types")
    return celltypist.train(sub, labels=label_key, feature_selection=True, use_SGD=True, mini_batch=True,
                            balance_cell_type=True, n_jobs=-1)


def main() -> None:
    args = parse_args(__doc__, DEFAULT_CONFIG)
    cfg = load_config(args.config)
    r = cfg["represent"]
    art, res = path(cfg["paths"]["artifacts"]), path(cfg["paths"]["results"])
    os.makedirs(art, exist_ok=True)
    os.makedirs(res, exist_ok=True)
    set_seed(cfg["train"]["seed"])

    adata = ad.read_h5ad(path(cfg["data"]["processed"]))
    batch_key = adata.uns.get("batch_key") or None

    print("[represent] 1/3 scVI")
    vae, genes = train_scvi(adata, batch_key, r)
    adata.obsm["X_scVI"] = vae.get_latent_representation()
    vae.save(os.path.join(art, "scvi"), overwrite=True)
    np.savez(os.path.join(art, "scvi_encoder.npz"), **export_scvi_encoder(vae))
    open(os.path.join(art, "genes.txt"), "w").write("\n".join(genes))

    print("[represent] 2/3 neighbours, UMAP, CellTypist on Perez labels")
    sc.pp.neighbors(adata, use_rep="X_scVI")
    sc.tl.umap(adata)
    ct_model = train_celltypist(adata, "cell_type_fine", cfg["labels"]["celltypist_max_cells"], cfg["train"]["seed"])
    np.savez(os.path.join(art, "celltypist.npz"), **export_celltypist(ct_model))

    print("[represent] 3/3 save")
    state = "disease_state" if "disease_state" in adata.obs else "disease"
    colors = ["cell_type_fine", state] + ([batch_key] if batch_key else [])
    fig = sc.pl.umap(adata, color=colors, ncols=1, frameon=False, size=2, legend_fontsize=6, return_fig=True)
    fig.savefig(os.path.join(res, "umap.png"), dpi=120, bbox_inches="tight")
    adata.write_h5ad(path(cfg["data"]["represented"]))
    print(f"[represent] wrote {cfg['data']['represented']} and {art}/scvi_encoder.npz, celltypist.npz, genes.txt")


if __name__ == "__main__":
    main()
