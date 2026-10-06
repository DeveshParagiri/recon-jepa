import math
import time
from pathlib import Path

import torch
from data import Observations, load_clean, order
from model import Experiment, beta_at
from utils import (
    digest,
    job_dir,
    object_digest,
    preserve_rng,
    read,
    restore_rng,
    rng_state,
    save,
    seed_all,
    tensor_digest,
)


def lr_at(index, settings):
    if index < settings["warmup"]:
        return settings["lr"] * index / settings["warmup"]
    return (
        settings["lr"]
        * (
            1
            + math.cos(
                math.pi
                * (index - settings["warmup"])
                / (settings["steps"] - settings["warmup"])
            )
        )
        / 2
    )


def gradient_diagnostic(experiment, raw, step):
    buffers = {k: v.clone() for k, v in experiment.named_buffers()}
    mode = experiment.training
    try:
        with preserve_rng():
            seed_all(1701)
            experiment.train()
            out = experiment(raw, step)
            terms = {
                "prediction": out["pred_loss"],
                "sigreg": 0.09 * out["sigreg_loss"],
                "jepa": out["pred_loss"] + 0.09 * out["sigreg_loss"],
                "reconstruction": beta_at(
                    experiment.job, step, experiment.protocol["training"]["steps"]
                )
                * out["recon_loss"],
            }
            norms = {}
            for name, loss in terms.items():
                gradients = torch.autograd.grad(
                    loss,
                    tuple(experiment.model.encoder.parameters()),
                    retain_graph=True,
                    allow_unused=True,
                )
                norms[name] = math.sqrt(
                    sum(
                        (
                            g.detach().double().square().sum().item()
                            for g in gradients
                            if g is not None
                        )
                    )
                )
            norms["reconstruction_over_jepa"] = norms["reconstruction"] / max(
                norms["jepa"], 1e-30
            )
            return dict(
                step=step,
                beta=beta_at(
                    experiment.job, step, experiment.protocol["training"]["steps"]
                ),
                norms=norms,
                losses={k: v.item() for k, v in terms.items()},
                batch_rule="same first training batch",
                mode="train; buffers and RNG restored",
            )
    finally:
        with torch.no_grad():
            for k, v in experiment.named_buffers():
                v.copy_(buffers[k])
        experiment.train(mode)


def create(root, protocol, job, device="cuda"):
    return Experiment(
        protocol,
        job,
        read(Path(root) / "data/normalization.json"),
        device,
        Path(root) / "official_source",
    )


def load_model(root, protocol, job, device="cuda"):
    saved = read(job_dir(root, job) / "checkpoint.pt")
    assert saved["step"] == protocol["training"]["steps"]
    experiment = create(root, protocol, job, device)
    experiment.load_state_dict(saved["model"])
    experiment.eval()
    return experiment


def train(root, job, device="cuda", on_save=lambda: None):
    root = Path(root)
    protocol = read(root / "protocol.json")
    settings = protocol["training"]
    directory = job_dir(root, job)
    if (directory / "train.done.json").exists():
        marker = read(directory / "train.done.json")
        assert marker["checkpoint_sha256"] == digest(directory / "checkpoint.pt")
        return marker
    assert not (root / "manifest.json").exists(), "Models are frozen."
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "job.json").exists():
        assert {
            k: v for k, v in read(directory / "job.json").items() if k != "studies"
        } == {k: v for k, v in job.items() if k != "studies"}
    else:
        save(directory / "job.json", job)
    clean = load_clean(root, "train")
    train_data = Observations(
        clean,
        protocol["data"]["train_noise_seed"],
        job["rho"],
        job["strength"],
        0,
        device,
        job.get("variant") == "weighted",
    )
    del clean
    total = settings["steps"]
    indices, sampler = order(
        job["seed"], len(train_data), total, settings["batch_size"], device
    )
    experiment = create(root, protocol, job, device)
    optimizer = torch.optim.AdamW(
        experiment.parameters(),
        lr=settings["lr"],
        weight_decay=settings["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda index: lr_at(index, settings) / settings["lr"]
    )
    seed_all(job["seed"])
    step0 = 0
    history = []
    elapsed_before = 0.0
    gradients = {}
    initial_sha = object_digest(
        {k: tensor_digest(v) for k, v in experiment.state_dict().items()}
    )
    checkpoint = directory / "checkpoint.pt"
    if checkpoint.exists():
        saved = read(checkpoint)
        assert (
            saved["protocol_sha256"] == object_digest(protocol)
            and {k: v for k, v in saved["job"].items() if k != "studies"}
            == {k: v for k, v in job.items() if k != "studies"}
            and (saved["sampler"]["sha256"] == sampler["sha256"])
        )
        assert saved["initial_sha256"] == initial_sha
        experiment.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        step0 = saved["step"]
        history = saved["history"]
        elapsed_before = saved["seconds"]
        gradients = saved["gradients"]
        restore_rng(saved["rng"])
        assert saved["sampler"]["cursor"] == step0
        del saved
    diagnose = (
        job["objective"] == "blend"
        and job["rho"] == 1
        and (job["seed"] == 3072)
        and (job.get("variant") is None)
    )
    if diagnose and "start" not in gradients:
        gradients["start"] = gradient_diagnostic(
            experiment, train_data.batch(indices[0]), 1
        )
    limit = total
    if device == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for step in range(step0 + 1, limit + 1):
        experiment.train()
        optimizer.zero_grad(set_to_none=True)
        out = experiment(train_data.batch(indices[step - 1]), step)
        assert torch.isfinite(out["loss"])
        out["loss"].backward()
        norm = torch.nn.utils.clip_grad_norm_(
            experiment.parameters(), settings["clip"], error_if_nonfinite=True
        )
        lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        values = {
            k: out[k].item()
            for k in (
                "loss",
                "pred_loss",
                "sigreg_loss",
                "recon_loss",
                "diff_sigreg_loss",
                "diff_pred_loss",
                "persistence_mse",
            )
        }
        history.append(
            dict(
                step=step,
                beta=beta_at(job, step, total),
                lr=lr,
                gradient_norm=norm.item(),
                **values,
            )
        )
        del out
        if step % settings["checkpoint_every"] == 0 or step == total:
            if step == total and diagnose:
                gradients["end"] = gradient_diagnostic(
                    experiment, train_data.batch(indices[0]), total
                )
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = elapsed_before + time.perf_counter() - started
            save(
                checkpoint,
                dict(
                    model={
                        k: v.detach().cpu() for k, v in experiment.state_dict().items()
                    },
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    step=step,
                    seconds=elapsed,
                    history=history,
                    gradients=gradients,
                    job=job,
                    protocol_sha256=object_digest(protocol),
                    initial_sha256=initial_sha,
                    rng=rng_state(),
                    sampler={**sampler, "cursor": step},
                ),
            )
            save(
                directory / "progress.json",
                dict(step=step, total=total, seconds=elapsed, updated=time.time()),
            )
            with preserve_rng():
                on_save()
            print(job["id"], step, "/", total, f"{elapsed / 60:.1f} min", flush=True)
    marker = dict(
        checkpoint_sha256=digest(checkpoint),
        seconds=read(checkpoint)["seconds"],
        step=total,
        protocol_sha256=object_digest(protocol),
    )
    save(directory / "train.done.json", marker)
    with preserve_rng():
        on_save()
    return marker
