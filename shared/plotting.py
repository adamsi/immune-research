"""Plot-ready summaries of attention weights."""
import numpy as np
import pandas as pd


def attention_by_group(att: np.ndarray, groups, top: int | None = None) -> pd.Series:
    """Sum attention weights per group (e.g. cell type), largest first.

    With `top`, groups beyond the first `top` are merged into a single "other" entry.
    """
    share = pd.Series(att).groupby(np.asarray(groups)).sum().sort_values(ascending=False)
    if top is not None and len(share) > top:
        share = pd.concat([share.iloc[:top], pd.Series({"other": share.iloc[top:].sum()})])
    return share
