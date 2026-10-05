"""Data loading helpers shared across projects: resumable downloads and dense float32 matrices."""
import os
import sys
import time
import urllib.request

import numpy as np


def download(url: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"  exists, skipping: {os.path.basename(dest)}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    t0 = time.time()

    def hook(blocks, bs, total):
        done = blocks * bs
        if total > 0 and blocks % 200 == 0:
            sys.stdout.write(f"\r  {os.path.basename(dest)}: {done / 1e6:,.0f} / {total / 1e6:,.0f} MB")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest + ".part", reporthook=hook)
    os.rename(dest + ".part", dest)
    print(f"\r  downloaded {os.path.basename(dest)} in {time.time() - t0:.0f}s")


def dense_float32(X) -> np.ndarray:
    """AnnData .X (sparse or dense) as a dense float32 array."""
    return X.toarray().astype(np.float32) if hasattr(X, "toarray") else np.asarray(X, dtype=np.float32)
