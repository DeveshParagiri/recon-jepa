import argparse
import copy
import itertools
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

BASE = Path(__file__).resolve().parent
METHODS = ["plain", "blend", "pixel", "motionjepa"]


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    if not re.fullmatch("[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", config["study"]):
        raise ValueError(
            "Study names must use letters, digits, hyphens or underscores."
        )
    if not 1 <= config["modal"]["concurrency"] <= 5:
        raise ValueError("Modal concurrency must be between 1 and 5.")
    protocol = config["protocol"]
    fixed = {
        "data": {
            "frames": 17,
            "image_size": 28,
            "dot_std": 1.3,
            "max_step": 4.0,
            "concentration": 0.2,
            "sigma": 0.03,
        },
        "training": {
            "precision": "float32",
            "tf32": False,
            "compile": False,
            "lambda_sigreg": 0.09,
        },
        "official": {"commit": "b3c2fb374b7c69f169de15774f371290a8736f56"},
    }
    for section, values in fixed.items():
        for key, expected in values.items():
            if protocol[section][key] != expected:
                raise ValueError(
                    f"This implementation requires {section}.{key}={expected!r}."
                )
    if protocol["seeds"] != [3072, 3073, 3074] or protocol["rhos"] != [
        1.0,
        0.99,
        0.9,
        0.5,
        0.0,
    ]:
        raise ValueError(
            "Selection and reports require the study's fixed seeds and correlations."
        )
    return config


def job_id(method, rho, seed, beta=0.0, variant=None, strength=0.03):
    speed = str(rho).replace(".", "p")
    b = str(float(beta)).replace(".", "p")
    return f"{variant or method}_rho{speed}_s{seed}_beta{b}_sigma{str(strength).replace('.', 'p')}"


def jobs(protocol, selected=None):
    result = {}

    def add(study, method, rho, seed, beta=0.0, variant=None, strength=0.03):
        key = job_id(method, rho, seed, beta, variant, strength)
        if key in result:
            if study not in result[key]["studies"]:
                result[key]["studies"].append(study)
        else:
            result[key] = dict(
                id=key,
                study=study,
                studies=[study],
                objective=method,
                rho=rho,
                seed=seed,
                beta=float(beta),
                variant=variant,
                strength=strength,
            )

    for seed in protocol["seeds"]:
        add("clean", "plain", 0.0, seed, strength=0.0)
    for method, rho, seed in itertools.product(
        METHODS, protocol["rhos"], protocol["seeds"]
    ):
        add("main", method, rho, seed, 300.0 if method == "blend" else 0.0)
    for beta in protocol["sweep_beta"]:
        add("sweep", "blend", 1.0, 3072, beta)
    for v, rho in itertools.product(protocol["variants"], [1.0, 0.0]):
        add("flicker_stage1", "blend", rho, 3072, 300.0, v)
    if selected:
        for rho, seed in itertools.product(protocol["rhos"], protocol["seeds"]):
            add("flicker_stage2", "blend", rho, seed, 300.0, selected)
    return list(result.values())


def select(root):
    from utils import job_dir, read, save

    root = Path(root)
    protocol = read(root / "protocol.json")
    initial = jobs(protocol)

    def score(method, rho, variant=None):
        j = next(
            (
                j
                for j in initial
                if j["objective"] == method
                and j["rho"] == rho
                and (j["seed"] == 3072)
                and (j["variant"] == variant)
                and (j["strength"] == 0.03)
                and (method != "blend" or j["beta"] == 300.0)
            )
        )
        return (
            read(job_dir(root, j) / "validation/metrics.json")["planning"]["CEM"][
                "success_rate"
            ]
            * 100
        )

    frozen = score("blend", 1.0)
    flicker = score("plain", 0.0)
    decisions = {}
    for v in protocol["variants"]:
        a, b = (score("blend", 1.0, v), score("blend", 0.0, v))
        decisions[v] = dict(
            frozen=a,
            flicker=b,
            passes=abs(a - frozen) <= 5 + 1e-10 and abs(b - flicker) <= 3 + 1e-10,
        )
    eligible = [v for v in protocol["variants"] if decisions[v]["passes"]]
    selected = (
        max(eligible, key=lambda v: decisions[v]["flicker"]) if eligible else None
    )
    result = dict(
        selected=selected,
        variants=decisions,
        split="validation",
        rule=protocol["criteria"]["stage1"],
    )
    if (root / "selection.json").exists():
        assert read(root / "selection.json") == result
    else:
        save(root / "selection.json", result)
    save(root / "queue.json", jobs(protocol, selected))
    return result


def execute(root, config, task, device="auto", on_save=lambda: None):
    import torch
    from data import prepare
    from utils import digest, job_dir, read, save

    root = Path(root)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu", "cuda"):
        raise ValueError("Use device: cpu or cuda.")
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    phase = task["phase"]
    if phase == "prepare":
        protocol = copy.deepcopy(config["protocol"])
        path = root / "protocol.json"
        if path.exists() and read(path) != protocol:
            raise ValueError("Settings changed; use a new study name.")
        save(path, protocol)
        selected = (
            read(root / "selection.json")["selected"]
            if (root / "selection.json").exists()
            else None
        )
        save(root / "queue.json", jobs(protocol, selected))
        prepare(root, protocol)
    elif phase in ("validation", "test"):
        from eval import evaluate
        from train import train

        job = task["job"]
        if phase == "validation":
            train(root, job, device, on_save)
        evaluate(root, job, phase, device, on_save)
    elif phase == "freeze":
        protocol = read(root / "protocol.json")
        queue = read(root / "queue.json")
        for job in queue:
            if not (job_dir(root, job) / "validation/eval.done.json").exists():
                raise RuntimeError("Run validation first: " + job["id"])
        differences = []
        for seed in protocol["seeds"]:
            scores = {}
            for method in ("plain", "motionjepa"):
                job = next(
                    j
                    for j in queue
                    if j["objective"] == method
                    and j["rho"] == 1.0
                    and j["seed"] == seed
                    and "main" in j["studies"]
                )
                scores[method] = read(job_dir(root, job) / "validation/metrics.json")[
                    "planning"
                ]["CEM"]["success_rate"]
            differences.append(100 * (scores["motionjepa"] - scores["plain"]))
        if min(differences) <= 0 or sum(differences) / len(differences) < 10 - 1e-10:
            raise RuntimeError(
                "MotionJEPA validation sanity check failed; inspect the method before testing."
            )
        manifest = {
            "models": [
                dict(
                    id=j["id"],
                    checkpoint_sha256=digest(job_dir(root, j) / "checkpoint.pt"),
                    probe_sha256=digest(job_dir(root, j) / "probe.pt"),
                )
                for j in queue
            ]
        }
        path = root / "manifest.json"
        if path.exists() and read(path) != manifest:
            raise ValueError("Frozen checkpoints or probes changed.")
        save(path, manifest)
        on_save()
        prepare(root, protocol, test=True)
    elif phase == "report":
        from plots import make, tables

        tables(root)
        make(root, device)
    else:
        raise ValueError("Unknown phase: " + phase)
    on_save()


def run(root, config, stage, dispatch, commit=lambda: None, reload=lambda: None):
    from utils import read

    root = Path(root)

    def batch(tasks):
        if tasks:
            dispatch(tasks)
            reload()

    if stage == "report":
        batch([{"phase": "report"}])
        return
    batch([{"phase": "prepare"}])
    protocol = read(root / "protocol.json")
    if stage == "pilot":
        job = next(
            j
            for j in jobs(protocol)
            if j["objective"] == "blend"
            and j["variant"] is None
            and j["rho"] == 1.0
            and j["seed"] == 3072
            and j["beta"] == 300.0
        )
        batch([{"phase": "validation", "job": job}])
        return
    if stage in ("all", "validation"):
        if not (root / "manifest.json").exists():
            batch(
                [{"phase": "validation", "job": j} for j in read(root / "queue.json")]
            )
            selected = select(root)["selected"]
            commit()
            if selected:
                batch(
                    [
                        {"phase": "validation", "job": j}
                        for j in read(root / "queue.json")
                        if "flicker_stage2" in j["studies"]
                    ]
                )
        if stage == "validation":
            return
    if not (root / "selection.json").exists():
        raise RuntimeError("Run validation before test evaluation.")
    batch([{"phase": "freeze"}])
    batch([{"phase": "test", "job": j} for j in read(root / "queue.json")])
    if stage == "all":
        batch([{"phase": "report"}])


def main():
    parser = argparse.ArgumentParser(description="Moving-dot world-model experiments")
    parser.add_argument("--config", default=str(BASE / "config.yaml"))
    parser.add_argument("--backend", choices=("local", "modal"), default="local")
    parser.add_argument(
        "--stage",
        choices=("all", "pilot", "validation", "test", "report"),
        default="all",
    )
    parser.add_argument("--study", help="Output subdirectory; defaults to config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.study:
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", args.study):
            parser.error("Invalid study name")
        config["study"] = args.study
    if args.backend == "modal":
        subprocess.run(
            [
                sys.executable,
                "-m",
                "modal",
                "run",
                "--detach",
                str(BASE / "modal_app.py"),
                "--stage",
                args.stage,
                "--study",
                config["study"],
            ],
            env=dict(os.environ, STUDY_CONFIG=str(Path(args.config).resolve())),
            check=True,
        )
    else:
        root = Path(config["output"]).resolve() / config["study"]

        def dispatch(tasks):
            for task in tasks:
                execute(root, config, task, config["device"])
            return True

        run(root, config, args.stage, dispatch)


if __name__ == "__main__":
    main()
