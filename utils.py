import contextlib
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as f:
        if path.suffix == ".json":
            f.write((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())
        else:
            torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def read(path):
    path = Path(path)
    return (
        json.loads(path.read_text())
        if path.suffix == ".json"
        else torch.load(path, map_location="cpu", weights_only=True)
    )


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def tensor_digest(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def object_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_state():
    return dict(
        cpu=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        python=random.getstate(),
        numpy=list(np.random.get_state()[:1])
        + [np.random.get_state()[1].tolist()]
        + list(np.random.get_state()[2:]),
    )


def restore_rng(state):
    torch.set_rng_state(state["cpu"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))


@contextlib.contextmanager
def preserve_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


def job_dir(root, job):
    return Path(root) / job["study"] / job["id"]
