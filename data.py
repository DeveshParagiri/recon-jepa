import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from utils import digest, read, save, tensor_digest
from vendor.slow_features.single import ContinuousMotionDataset

IMAGE_SCALE = 2 * math.pi * 1.3**2


def simulator(batch_size=1, **kw):
    return ContinuousMotionDataset(
        size=1,
        batch_size=batch_size,
        img_size=28,
        std=1.3,
        noise=0.0,
        static_noise=0.0,
        normalize=False,
        device=torch.device("cpu"),
        **kw,
    )


def generate(count, seed, frames=17, batch_size=64):
    chunks = {k: [] for k in ("images", "actions", "positions")}
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        for offset in range(0, count, batch_size):
            sample = simulator(
                min(batch_size, count - offset),
                n_steps=frames,
                max_step=4.0,
                concentration=0.2,
            ).generate_multistep_sample()
            chunks["images"].append(sample.states)
            chunks["actions"].append(sample.actions[:, :, 0])
            chunks["positions"].append(sample.locations[:, :, 0])
    return {k: torch.cat(v) for k, v in chunks.items()}


def fields(count, frames, seed, rho):
    g = torch.Generator().manual_seed(seed)
    result = torch.empty(count, frames, 1, 28, 28)
    for t in range(frames):
        noise = torch.randn(count, 1, 28, 28, generator=g)
        result[:, t] = (
            noise
            if t == 0 or rho == 0
            else result[:, 0]
            if rho == 1
            else rho * result[:, t - 1] + math.sqrt(1 - rho * rho) * noise
        )
    return result


def prepare(root, protocol, test=False):
    root = Path(root)
    folder = root / "data"
    folder.mkdir(exist_ok=True)
    names = ["test"] if test else ["train", "validation"]
    spec = protocol["data"]
    if test:
        assert (root / "manifest.json").exists(), (
            "Freeze every model before generating the test set."
        )
    for name in names:
        path = folder / f"{name}.pt"
        meta = folder / f"{name}.json"
        expected = dict(
            count=spec[name],
            seed=spec[name + "_seed"],
            frames=spec["frames"],
            generation_batch=spec["generation_batch"],
        )
        if meta.exists():
            m = read(meta)
            assert m["spec"] == expected and digest(path) == m["sha256"]
            continue
        clean = generate(
            expected["count"],
            expected["seed"],
            expected["frames"],
            expected["generation_batch"],
        )
        save(path, clean)
        save(
            meta,
            dict(
                spec=expected,
                sha256=digest(path),
                tensor_sha256={k: tensor_digest(v) for k, v in clean.items()},
            ),
        )
        print("Generated", name, expected["count"], "clips", flush=True)
    if not test:
        clean = read(folder / "train.pt")
        a = clean["actions"].flatten(0, 1)
        normal = dict(mean=a.mean(0).tolist(), std=a.std(0).tolist())
        if (folder / "normalization.json").exists():
            assert read(folder / "normalization.json") == normal
        else:
            save(folder / "normalization.json", normal)
    return {name: read(folder / f"{name}.json")["sha256"] for name in names}


def load_clean(root, split):
    source = Path(root) / "data" / f"{split}.pt"
    assert digest(source) == read(source.with_suffix(".json"))["sha256"]
    return read(source)


class Observations:
    def __init__(
        self,
        clean,
        noise_seed,
        rho,
        strength,
        planning_budget=20,
        device="cpu",
        keep_clean=False,
    ):
        self.strength = strength
        self.rho = rho
        f = fields(len(clean["images"]), max(17, planning_budget + 3), noise_seed, rho)
        self.images = (clean["images"] + strength / IMAGE_SCALE * f[:, :17]).to(device)
        self.actions = clean["actions"].to(device)
        self.positions = clean["positions"]
        self.fields = f if str(device) == "cpu" else None
        self.clean = (
            clean
            if str(device) == "cpu"
            else {"images": clean["images"].to(device)}
            if keep_clean
            else None
        )

    def observe(self, images, ids, frame):
        return images + self.strength / IMAGE_SCALE * self.fields[ids, frame][:, None]

    def batch(self, ids):
        trajectory = ids // 14
        start = ids % 14
        x = self.images[
            trajectory[:, None], start[:, None] + torch.arange(4, device=ids.device)
        ]
        a = self.actions[
            trajectory[:, None], start[:, None] + torch.arange(3, device=ids.device)
        ]
        c = (
            None
            if self.clean is None
            else self.clean["images"][
                trajectory[:, None], start[:, None] + torch.arange(4, device=ids.device)
            ]
        )
        return (x, a, c)

    def __len__(self):
        return len(self.images) * 14


def order(seed, n, steps, batch_size, device):
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        range(n),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=g,
        num_workers=0,
    )
    iterator = iter(loader)
    indices = []
    for _ in range(steps):
        try:
            indices.append(next(iterator))
        except StopIteration:
            iterator = iter(loader)
            indices.append(next(iterator))
    all_ids = torch.stack(indices)
    return (
        all_ids.to(device),
        dict(seed=seed, sha256=tensor_digest(all_ids), generator_state=g.get_state()),
    )
