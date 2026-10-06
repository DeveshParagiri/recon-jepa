import time
from pathlib import Path

import torch
from data import IMAGE_SCALE, Observations, load_clean, simulator
from planning import goal_background_check, plan
from train import load_model
from utils import digest, job_dir, object_digest, preserve_rng, read, save, seed_all


@torch.inference_mode()
def embeddings(experiment, images, indices=None):
    experiment.eval()
    if indices is None:
        indices = torch.arange(len(images))
    return torch.cat(
        [experiment.encode(images[ids, None])[:, 0].cpu() for ids in indices.split(256)]
    )


def fit(features, targets, ridge):
    x, y = (features.double(), targets.double())
    mean = x.mean(0)
    std = x.std(0).clamp_min(1e-06)
    target_mean = y.mean(0)
    x = (x - mean) / std
    weights = torch.linalg.solve(
        x.T @ x / len(x) + ridge * torch.eye(x.shape[1], dtype=x.dtype),
        x.T @ (y - target_mean) / len(x),
    )
    return dict(mean=mean, std=std, target_mean=target_mean, weights=weights)


def predict(features, probe):
    return (
        (features.double() - probe["mean"]) / probe["std"] @ probe["weights"]
        + probe["target_mean"]
    ).float()


def errors(prediction, target):
    delta = prediction - target
    return dict(
        mean_distance_px=delta.norm(dim=-1).mean().item(),
        rmse_per_coordinate_px=delta.square().mean().sqrt().item(),
    )


def fit_probe(experiment, training, path, checkpoint_hash):
    if path.exists():
        result = read(path)
        assert result["checkpoint_sha256"] == checkpoint_hash
        return result
    cfg = experiment.protocol["probe"]
    ids = torch.randperm(
        len(training.images) * 17,
        generator=torch.Generator().manual_seed(cfg["index_seed"]),
    )[: cfg["frames"]]
    features = embeddings(experiment, training.images.flatten(0, 1), ids)
    target = training.positions.flatten(0, 1)[ids]
    result = dict(
        fit=fit(features, target, cfg["ridge"]),
        indices=ids,
        ridge=cfg["ridge"],
        split="train",
        checkpoint_sha256=checkpoint_hash,
    )
    save(path, result)
    return result


def nearest(query, gallery):
    gallery = gallery.double()
    choices = []
    margins = []
    for q in query.double().split(256):
        distances = (
            q.square().sum(1, keepdim=True)
            + gallery.square().sum(1)[None]
            - 2 * q @ gallery.T
        )
        best = distances.topk(2, largest=False, dim=1)
        choices.append(best.indices[:, 0])
        margins.append(best.values[:, 1] - best.values[:, 0])
    return (torch.cat(choices), torch.cat(margins))


def retrieval_score(query, gallery):
    ids, margins = nearest(query, gallery)
    return dict(
        correct=int((ids == torch.arange(len(ids))).sum()),
        queries=len(ids),
        top1_accuracy=(ids == torch.arange(len(ids))).double().mean().item(),
        chance_accuracy=1 / len(ids),
        exact_distance_ties=int((margins == 0).sum()),
    )


def summarize_prediction(saved):
    totals = {k: float(v.double().mean()) for k, v in saved["per_window"].items()}
    totals["prediction_to_no_change"] = totals["pred_loss"] / max(
        totals["persistence_mse"], 1e-12
    )
    return totals


@torch.inference_mode()
def predictions(experiment, observations, ids):
    terms = {k: [] for k in ("pred_loss", "persistence_mse")}
    mode = experiment.training
    predicted = []
    targets = []
    persistent = []
    with preserve_rng():
        seed_all(1701)
        experiment.eval()
        for group in ids.split(128):
            out = experiment(
                observations.batch(group.to(observations.images.device)),
                return_predictions=True,
            )
            p, t, b = (out[k].cpu() for k in ("predicted", "target", "persistent"))
            predicted.append(p)
            targets.append(t)
            persistent.append(b)
            terms["pred_loss"].append((p - t).flatten(1).square().mean(1))
            terms["persistence_mse"].append((b - t).flatten(1).square().mean(1))
        experiment.train(mode)
    return dict(
        ids=ids,
        predicted=torch.cat(predicted),
        target=torch.cat(targets),
        persistent=torch.cat(persistent),
        per_window={k: torch.cat(v) for k, v in terms.items()},
    )


@torch.inference_mode()
def shuffled_predictions(experiment, observations, ids):
    ps = []
    ts = []
    with preserve_rng():
        seed_all(1701)
        experiment.eval()
        for group in ids.split(128):
            out = experiment(
                observations.batch(group.to(observations.images.device)),
                shuffle=True,
                return_predictions=True,
            )
            ps.append(out["predicted"].cpu())
            ts.append(out["target"].cpu())
    p, t = (torch.cat(ps), torch.cat(ts))
    return dict(
        ids=ids,
        predicted=p,
        target=t,
        per_window_mse=(p - t).flatten(1).square().mean(1),
    )


def retrieval(experiment, data):
    pairs = {}
    gallery = embeddings(experiment, data.images[:, 0])
    for frame in (8, 16):
        pairs[f"frame_{frame}_to_0"] = dict(
            query=embeddings(experiment, data.images[:, frame]), gallery=gallery
        )
    field = data.strength / IMAGE_SCALE * data.fields[:, 0]
    pairs["shared_frozen"] = dict(
        query=embeddings(experiment, data.clean["images"][:, 16] + field),
        gallery=embeddings(experiment, data.clean["images"][:, 0] + field),
    )
    env = simulator()
    positions = [
        torch.tensor(xy).expand(len(field), 2) for xy in ([7.0, 7.0], [20.0, 20.0])
    ]
    control = [
        embeddings(experiment, env.render_location(p)[:, None] + field)
        for p in positions
    ]
    pairs["controlled_positions"] = dict(query=control[1], gallery=control[0])
    return pairs


def goal_score(value):
    d = (value["actions"]["original"] - value["actions"]["swapped"]).norm(dim=-1)
    return dict(
        mean_action_change=d.mean().item(),
        rms_action_change=d.square().mean().sqrt().item(),
        changed_fraction=(d > value["threshold"]).float().mean().item(),
        trials=len(d),
    )


def evaluate(root, job, split="validation", device="cuda", on_save=lambda: None):
    root = Path(root)
    protocol = read(root / "protocol.json")
    directory = job_dir(root, job)
    outdir = directory / split
    outdir.mkdir(parents=True, exist_ok=True)
    if split == "test":
        manifest = read(root / "manifest.json")
        entry = next((x for x in manifest["models"] if x["id"] == job["id"]))
        assert entry["checkpoint_sha256"] == digest(
            directory / "checkpoint.pt"
        ) and entry["probe_sha256"] == digest(directory / "probe.pt")
    sources = dict(
        checkpoint_sha256=digest(directory / "checkpoint.pt"),
        data_sha256=read(root / "data" / f"{split}.json")["sha256"],
        protocol_sha256=object_digest(protocol),
    )
    if (outdir / "source.json").exists():
        assert read(outdir / "source.json") == sources
    else:
        save(outdir / "source.json", sources)
    if (outdir / "eval.done.json").exists():
        marker = read(outdir / "eval.done.json")
        assert digest(outdir / "metrics.json") == marker["metrics_sha256"]
        return read(outdir / "metrics.json")
    experiment = load_model(root, protocol, job, device)
    experiment.config["planning"] = dict(protocol["planning"])
    if split == "test":
        experiment.config["planning"]["trials"] = protocol["planning"]["test_trials"]
    if not (directory / "probe.pt").exists():
        assert split == "validation", "Fit probes before the test manifest."
        training = Observations(
            load_clean(root, "train"),
            protocol["data"]["train_noise_seed"],
            job["rho"],
            job["strength"],
            0,
        )
        tick = time.perf_counter()
        fit_probe(
            experiment, training, directory / "probe.pt", sources["checkpoint_sha256"]
        )
        save(directory / "probe_time.json", dict(seconds=time.perf_counter() - tick))
        del training
        with preserve_rng():
            on_save()
    probe = read(directory / "probe.pt")
    assert probe["checkpoint_sha256"] == sources["checkpoint_sha256"]
    data = Observations(
        load_clean(root, split),
        protocol["data"][split + "_noise_seed"],
        job["rho"],
        job["strength"],
    )

    def component(name, compute):
        path = outdir / (name + ".pt")
        if not path.exists():
            tick = time.perf_counter()
            value = compute()
            if device == "cuda":
                torch.cuda.synchronize()
            save(path, dict(value=value, seconds=time.perf_counter() - tick))
            with preserve_rng():
                on_save()
        return read(path)["value"]

    ids = torch.arange(len(data))
    subset = torch.randperm(len(data), generator=torch.Generator().manual_seed(1701))[
        :1024
    ]
    full = component("prediction", lambda: predictions(experiment, data, ids))
    full_score = summarize_prediction(full)
    del full
    normal = component("subset", lambda: predictions(experiment, data, subset))
    normal_score = summarize_prediction(normal)
    del normal
    shuffled = component(
        "shuffled", lambda: shuffled_predictions(experiment, data, subset)
    )
    shuffle_mse = shuffled["per_window_mse"].mean().item()
    del shuffled

    def positions():
        target = data.positions.flatten(0, 1)
        features = embeddings(experiment, data.images.flatten(0, 1))
        return dict(
            prediction=predict(features, probe["fit"]),
            target=target,
            probe_sha256=digest(directory / "probe.pt"),
            fit_split="train",
        )

    position = component("position", positions)
    position_score = errors(position["prediction"], position["target"])
    del position
    retrieved = component("retrieval", lambda: retrieval(experiment, data))
    retrieval_scores = {
        k: retrieval_score(v["query"], v["gallery"]) for k, v in retrieved.items()
    }
    del retrieved
    planning = component("planning", lambda: plan(experiment, data))
    swapped = component(
        "goal_background",
        lambda: goal_background_check(experiment, data, planning["trajectory_ids"]),
    )
    metrics = dict(
        id=job["id"],
        job=job,
        split=split,
        planning=planning["results"],
        position=position_score,
        retrieval=retrieval_scores,
        prediction=full_score,
        action_shuffle_ratio=shuffle_mse / max(normal_score["pred_loss"], 1e-12),
        goal_background=goal_score(swapped),
    )
    save(outdir / "metrics.json", metrics)
    artifacts = {p.name: digest(p) for p in outdir.glob("*.pt")}
    save(
        outdir / "eval.done.json",
        dict(
            metrics_sha256=digest(outdir / "metrics.json"),
            artifacts=artifacts,
            seconds=sum((read(outdir / p)["seconds"] for p in artifacts)),
        ),
    )
    with preserve_rng():
        on_save()
    print(
        "EVALUATED",
        job["id"],
        split,
        metrics["planning"]["CEM"]["success_rate"],
        flush=True,
    )
    return metrics
