"""Hash numerical contents rather than ZIP-container timestamps."""
import hashlib
from pathlib import Path
import numpy as np


def content_hash(path):
    path=Path(path)
    digest=hashlib.sha256()
    if path.suffix=='.npz':
        with np.load(path,allow_pickle=False) as arrays:
            for name in sorted(arrays.files):
                value=arrays[name]
                digest.update(name.encode())
                digest.update(str(value.dtype).encode())
                digest.update(str(value.shape).encode())
                digest.update(value.tobytes(order='C'))
    else:
        digest.update(path.read_bytes())
    return digest.hexdigest()
