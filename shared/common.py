"""Config loading, seeding, and turning a cell matrix into patient bags."""
import argparse
import random

import numpy as np
import yaml


def parse_args(description: str, default_config: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=default_config)
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def build_bags(obs, Z, patient_col="Patient_ID", celltype_col="annotated_clusters"):
    """Group cell embeddings by patient.

    obs: DataFrame with patient id, y, cell type (one row per cell, aligned with Z)
    Z:   (n_cells, n_pcs) float32
    Returns patients (sorted array), bags {patient: (N_i, n_pcs)}, y (n_patients,), ctype {patient: (N_i,)}
    """
    pat = obs[patient_col].values
    patients = np.unique(pat)
    bags = {p: Z[pat == p] for p in patients}
    ctype = {p: obs.loc[pat == p, celltype_col].values for p in patients}
    y = np.array([int(obs.loc[pat == p, "y"].iloc[0]) for p in patients])
    return patients, bags, y, ctype
