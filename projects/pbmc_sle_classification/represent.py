"""Step 2: scVI latent space (batch removed), CellTypist cell types, UMAP atlas.

Input : data/processed/pbmc.h5ad
Output: data/processed/pbmc_rep.h5ad   + obsm["X_scVI"], obs["cell_type_pred"], obsm["X_umap"]
        artifacts/genes.txt            the HVGs scVI reads, in order
        artifacts/scvi_encoder.npz     scVI encoder weights as numpy (the app embeds new patients with it)
        artifacts/celltypist.npz       CellTypist classifier as numpy (the app labels new cells with it)
        artifacts/scvi/                full scVI model (for reuse in notebooks)
        results/umap.png               atlas coloured by cell type, disease and batch
"""
import gc
import os

import anndata as ad
import matplotlib
import numpy as np
import scanpy as sc

from shared.common import load_config, parse_args, set_seed

from . import DEFAULT_CONFIG, path
from .features import export_celltypist, export_scvi_encoder

matplotlib.use("Agg")


def train_scvi(adata: ad.AnnData, batch_key: str | None, n_latent: int, epochs):
    """scVI on raw counts of the HVGs. batch_key goes to the decoder only (encode_covariates=False),
    so the encoder is batch-free and can embed a new patient from an unseen batch."""
    import scvi

    hvg = adata[:, adata.var["highly_variable"]].copy()
    scvi.model.SCVI.setup_anndata(hvg, layer="counts", batch_key=batch_key)
    vae = scvi.model.SCVI(hvg, n_latent=n_latent, encode_covariates=False)
    vae.train(max_epochs=epochs)
    return vae, hvg.var_names.tolist()


def annotate(adata: ad.AnnData, model_name: str, chunk: int = 20_000):
    """CellTypist on log-normalised data (all genes), majority voting over fine Leiden clusters of the scVI graph."""
    import celltypist
    from celltypist import models

    models.download_models(model=model_name)
    model = models.Model.load(model_name)
    sc.tl.leiden(adata, resolution=3, key_added="over_clust", flavor="igraph", n_iterations=2)
    # CellTypist builds a dense cells x genes matrix, so predict in chunks to keep RAM low (fits 12 GB Colab)
    labels = np.empty(adata.n_obs, dtype=object)
    for start in range(0, adata.n_obs, chunk):
        part = adata[start:start + chunk].copy()
        labels[start:start + chunk] = celltypist.annotate(part, model=model).predicted_labels["predicted_labels"].astype(str).values
        del part
        gc.collect()
    adata.obs["cell_type_raw"] = labels
    # majority voting, as celltypist does it: each fine cluster takes its most common label
    adata.obs["cell_type_pred"] = (adata.obs.groupby("over_clust", observed=True)["cell_type_raw"]
                                   .transform(lambda s: s.value_counts().index[0]).astype(str))
    return model


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
    vae, genes = train_scvi(adata, batch_key, r["n_latent"], r["scvi_epochs"])
    adata.obsm["X_scVI"] = vae.get_latent_representation()
    vae.save(os.path.join(art, "scvi"), overwrite=True)
    np.savez(os.path.join(art, "scvi_encoder.npz"), **export_scvi_encoder(vae))
    open(os.path.join(art, "genes.txt"), "w").write("\n".join(genes))

    print("[represent] 2/3 neighbours, UMAP, CellTypist")
    sc.pp.neighbors(adata, use_rep="X_scVI")
    sc.tl.umap(adata)
    ct_model = annotate(adata, r["celltypist_model"])
    np.savez(os.path.join(art, "celltypist.npz"), **export_celltypist(ct_model))
    print(adata.obs["cell_type_pred"].value_counts().head(15))

    print("[represent] 3/3 save")
    colors = ["cell_type_pred", "disease"] + ([batch_key] if batch_key else [])
    fig = sc.pl.umap(adata, color=colors, ncols=1, frameon=False, size=2, legend_fontsize=6, return_fig=True)
    fig.savefig(os.path.join(res, "umap.png"), dpi=120, bbox_inches="tight")
    adata.write_h5ad(path(cfg["data"]["represented"]))
    print(f"[represent] wrote {cfg['data']['represented']} and {art}/scvi_encoder.npz, celltypist.npz, genes.txt")


if __name__ == "__main__":
    main()
