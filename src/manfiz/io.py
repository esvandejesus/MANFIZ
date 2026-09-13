"""Portable JSON and array fingerprints; no MATLAB or pickle dependency."""
from dataclasses import asdict, is_dataclass
from pathlib import Path
import hashlib
import json
import math
import numpy as np


def jsonable(value):
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False)+"\n", encoding="utf-8")


def fingerprint(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        a = np.ascontiguousarray(value, dtype='<f8')
        digest.update(str(a.shape).encode('ascii'))
        digest.update(a.tobytes())
    return digest.hexdigest()
