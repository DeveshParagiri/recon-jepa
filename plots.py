import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from data import IMAGE_SCALE, Observations, load_clean, simulator
from PIL import Image
from train import load_model
from utils import job_dir, read, save

plt.switch_backend("Agg")

LABELS = {
    "plain": "LeWM",
    "blend": "LeWM + reconstruction",
    "pixel": "Pixel model",
    "motionjepa": "MotionJEPA",
    "blur": "Blurred target",
    "weighted": "Dot-weighted (privileged)",
    "decay": "Decaying beta",
}


def collect(root, split="test"):
    result = []
    for job in read(Path(root) / "queue.json"):
        m = read(job_dir(root, job) / split / "metrics.json")
        cem = m["planning"]["CEM"]
        interval = cem["wilson_95"]
        result.append(
            dict(
                id=job["id"],
                studies=job["studies"],
                model=job["variant"] or job["objective"],
                rho=job["rho"],
                sigma=job["strength"],
                seed=job["seed"],
                beta=job["beta"],
                planning_percent=100 * cem["success_rate"],
                successes=cem["successes"],
                trials=cem["trials"],
                wilson_low=100 * interval[0],
                wilson_high=100 * interval[1],
                position_px=m["position"]["mean_distance_px"],
                retrieval_percent=100
                * m["retrieval"]["frame_16_to_0"]["top1_accuracy"],
                controlled_retrieval_percent=100
                * m["retrieval"]["controlled_positions"]["top1_accuracy"],
                prediction_ratio=m["prediction"]["prediction_to_no_change"],
                action_shuffle_ratio=m["action_shuffle_ratio"],
                goal_swap_action_rms=m["goal_background"]["rms_action_change"],
                goal_swap_changed_percent=100
                * m["goal_background"]["changed_fraction"],
                training_seconds=read(job_dir(root, job) / "train.done.json")[
                    "seconds"
                ],
            )
        )
    return result


def criteria(root, rows):

    def get(model, rho, seed):
        return next(
            (
                r
                for r in rows
                if "main" in r["studies"]
                and r["model"] == model
                and (r["rho"] == rho)
                and (r["seed"] == seed)
            )
        )

    paired = {}
    main = True
    for rho in (1.0, 0.99, 0.0):
        for seed in (3072, 3073, 3074):
            diff = (
                get("blend", rho, seed)["planning_percent"]
                - get("plain", rho, seed)["planning_percent"]
            )
            passed = abs(diff) <= 10 + 1e-10 if rho == 0 else diff >= 20 - 1e-10
            paired[f"rho{rho}_seed{seed}"] = dict(difference_points=diff, passed=passed)
            main &= passed
    motion = [
        get("motionjepa", 1.0, s)["planning_percent"]
        - get("plain", 1.0, s)["planning_percent"]
        for s in (3072, 3073, 3074)
    ]
    return dict(
        main_passed=main,
        paired=paired,
        motion_sanity_passed=statistics.mean(motion) >= 10
        and all((d > 0 for d in motion)),
        motion_frozen_differences=motion,
        stage1=read(Path(root) / "selection.json"),
    )


def tables(root):
    root = Path(root)
    rows = collect(root)
    folder = root / "tables"
    folder.mkdir(exist_ok=True)
    for row in rows:
        row["privileged_training_information"] = row["model"] == "weighted"
        row["analysis_role"] = (
            "supporting"
            if row["model"] == "weighted"
            else "exploratory"
            if row["model"] in ("blur", "decay")
            else "main_or_reference"
        )

    def write(name, values):
        if not values:
            return
        with (folder / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)

    write("all.csv", rows)
    for study in ("clean", "main", "sweep", "flicker_stage1", "flicker_stage2"):
        selected = [r for r in rows if study in r["studies"]]
        write(study + ".csv", selected)
        groups = {}
        for row in selected:
            groups.setdefault((row["model"], row["rho"], row["beta"]), []).append(row)
        means = [
            dict(
                model=kind,
                rho=rho,
                beta=beta,
                seeds=len(values),
                planning_mean=statistics.mean((v["planning_percent"] for v in values)),
                planning_min=min((v["planning_percent"] for v in values)),
                planning_max=max((v["planning_percent"] for v in values)),
                position_mean=statistics.mean((v["position_px"] for v in values)),
            )
            for (kind, rho, beta), values in groups.items()
        ]
        write(study + "_means.csv", means)
    gradients = []
    for job in read(root / "queue.json"):
        if (
            job["objective"] == "blend"
            and job["rho"] == 1.0
            and (job["seed"] == 3072)
            and (job["variant"] is None)
        ):
            for phase, values in read(job_dir(root, job) / "checkpoint.pt")[
                "gradients"
            ].items():
                gradients.append(dict(beta=job["beta"], phase=phase, **values["norms"]))
    write("encoder_gradient_diagnostics.csv", gradients)
    first = read(root / "queue.json")[0]
    controls = read(job_dir(root, first) / "test/metrics.json")["planning"]
    write(
        "control_baselines.csv",
        [dict(policy=k, **v) for k, v in controls.items() if k != "CEM"],
    )
    save(root / "criteria.json", criteria(root, rows))
    return rows


"Exact, CPU-resident inputs for every figure; cache before rendering."


def prepare_inputs(root, device="cuda"):
    root = Path(root)
    folder = root / "figures"
    folder.mkdir(exist_ok=True)
    path = folder / "inputs.pt"
    if path.exists():
        return read(path)
    p = read(root / "protocol.json")
    queue = read(root / "queue.json")

    def job(kind, rho=1.0, seed=3072, beta=None, variant=None):
        return next(
            (
                j
                for j in queue
                if j["objective"] == kind
                and j["variant"] == variant
                and (j["rho"] == rho)
                and (j["seed"] == seed)
                and (j["strength"] == 0.03)
                and (beta is None or j["beta"] == beta)
            )
        )

    def decode(j, images):
        model = load_model(root, p, j, device)
        with torch.inference_mode():
            result = model.decoder(model.encode(images)).cpu().reshape(6, 28, 28)
        return result

    planning = {
        j["id"]: read(job_dir(root, j) / "test/planning.pt")["value"]
        for j in queue
        if j["objective"] in ("plain", "blend")
        and j["variant"] is None
        and (j["rho"] == 1.0)
        and (j["study"] == "main")
    }
    validation = load_clean(root, "validation")
    val = Observations(validation, p["data"]["validation_noise_seed"], 1.0, 0.03)
    images = val.images[:6, :1]
    reconstruction = [("Input", images[:, 0, 0] * IMAGE_SCALE)] + [
        (f"β = {b:g}", decode(job("blend", beta=b), images)) for b in (1.0, 300.0)
    ]
    variants = {}
    for rho, condition in ((1.0, "frozen"), (0.0, "flicker")):
        obs = Observations(validation, p["data"]["validation_noise_seed"], rho, 0.03)
        images = obs.images[:6, :1]
        variants[condition] = [("Input", images[:, 0, 0] * IMAGE_SCALE)]
        for variant in p["variants"]:
            variants[condition].append(
                (
                    LABELS[variant],
                    decode(job("blend", rho=rho, variant=variant), images),
                )
            )
    backgrounds = []
    for rho in (1.0, 0.99, 0.0):
        obs = Observations(validation, p["data"]["validation_noise_seed"], rho, 0.03)
        backgrounds.append(
            validation["images"][0, :, 0] * IMAGE_SCALE
            + 0.03 * 4 * obs.fields[0, :17, 0]
        )
    left = planning[job("plain")["id"]]
    right = planning[job("blend", beta=300.0)["id"]]
    assert torch.equal(left["trajectory_ids"], right["trajectory_ids"])
    distance = (
        (left["paths"]["CEM"][:, -1] - left["goals"]).norm(dim=-1)
        + (right["paths"]["CEM"][:, -1] - right["goals"]).norm(dim=-1)
    ) / 2
    ranked = sorted(
        range(len(distance)),
        key=lambda i: (float(distance[i]), int(left["trajectory_ids"][i])),
    )
    selected = ranked[(len(ranked) - 1) // 2]
    tid = int(left["trajectory_ids"][selected])
    obs = Observations(
        load_clean(root, "test"), p["data"]["test_noise_seed"], 1.0, 0.03
    )
    env = simulator()
    hero_images = torch.stack(
        [
            env.render_location(q["paths"]["CEM"][selected]) * IMAGE_SCALE
            + 0.03 * obs.fields[tid, 0, 0]
            for q in (left, right)
        ]
    )
    selection = dict(
        seed=3072,
        trajectory_id=tid,
        index=selected,
        rule=p["figures"]["hero"],
        paired_final_distance=distance[selected].item(),
    )
    value = dict(
        rows=collect(root),
        planning=planning,
        reconstruction=reconstruction,
        variants=variants,
        backgrounds=torch.stack(backgrounds),
        hero_images=hero_images,
        hero_selection=selection,
        reconstruction_trajectory_ids=torch.arange(6),
        reconstruction_frame=0,
        background_trajectory_id=0,
        background_contrast=4.0,
    )
    save(path, value)
    return value


COLORS = {
    "plain": "#3366AA",
    "blend": "#DD6633",
    "pixel": "#779944",
    "motionjepa": "#8866AA",
    "blur": "#22AA99",
    "weighted": "#BB5599",
    "decay": "#AA9944",
}
SPEEDS = [1.0, 0.99, 0.9, 0.5, 0.0]
SPEED_LABELS = ["Frozen", "Slow drift", "ρ = 0.9", "ρ = 0.5", "Flicker"]


def export(fig, path):
    fig.tight_layout()
    fig.savefig(path.with_suffix(".svg"))
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def frame(fig):
    fig.canvas.draw()
    array = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return Image.fromarray(array)


def gif(images, path):
    images[0].save(path, save_all=True, append_images=images[1:], duration=200, loop=0)


def make(root, device="cuda"):
    root = Path(root)
    folder = root / "figures"
    folder.mkdir(exist_ok=True)
    queue = read(root / "queue.json")
    saved = prepare_inputs(root, device)
    rows = saved["rows"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 100,
        }
    )

    def job(kind, rho=1.0, seed=3072, beta=None):
        return next(
            (
                j
                for j in queue
                if j["objective"] == kind
                and j["variant"] is None
                and (j["rho"] == rho)
                and (j["seed"] == seed)
                and (j["strength"] == 0.03)
                and (beta is None or j["beta"] == beta)
            )
        )

    def component(j, name):
        return saved["planning"][j["id"]]

    fig, ax = plt.subplots(figsize=(9, 4))
    for index, kind in enumerate(("plain", "blend", "pixel", "motionjepa")):
        vals = [
            [
                r["planning_percent"]
                for r in rows
                if "main" in r["studies"] and r["model"] == kind and (r["rho"] == rho)
            ]
            for rho in SPEEDS
        ]
        x = np.arange(5) + (index - 1.5) * 0.18
        ax.plot(
            x,
            [np.mean(v) for v in vals],
            color=COLORS[kind],
            label=LABELS[kind],
            lw=1.7,
        )
        for i, v in enumerate(vals):
            ax.scatter(
                x[i] + np.linspace(-0.035, 0.035, len(v)), v, color=COLORS[kind], s=23
            )
    ax.set(
        xticks=range(5),
        xticklabels=SPEED_LABELS,
        ylim=(0, 100),
        ylabel="Planning success (%)",
    )
    ax.legend(fontsize=8, ncol=2)
    export(fig, folder / "main_results")
    fig, ax = plt.subplots(figsize=(8, 4))
    for kind in ("plain", "blend", "pixel", "motionjepa"):
        means = []
        for rho in SPEEDS:
            vals = [
                r["planning_percent"]
                for r in rows
                if "main" in r["studies"] and r["model"] == kind and (r["rho"] == rho)
            ]
            means.append(np.mean(vals))
            ax.scatter([rho] * len(vals), vals, color=COLORS[kind], s=18, alpha=0.65)
        ax.plot(SPEEDS, means, "o-", color=COLORS[kind], label=LABELS[kind])
    ax.set(
        xlim=(1.02, -0.02),
        ylim=(0, 100),
        xlabel="Background persistence ρ (frozen → flicker)",
        ylabel="Planning success (%)",
    )
    ax.legend(fontsize=8)
    export(fig, folder / "speed_curve")
    sweep = sorted(
        [r for r in rows if "sweep" in r["studies"]], key=lambda x: x["beta"]
    )
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, key, label, limits in zip(
        axs,
        ["planning_percent", "position_px"],
        ["Planning success (%)", "Position error (px)"],
        [(0, 100), (0, 20)],
    ):
        ax.plot(
            [r["beta"] for r in sweep],
            [r[key] for r in sweep],
            "o-",
            color=COLORS["blend"],
        )
        ax.set(
            xscale="log", xlabel="Reconstruction weight β", ylabel=label, ylim=limits
        )
    export(fig, folder / "beta_sweep")
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    for ax, kind in zip(axs, ("plain", "blend")):
        for seed in (3072, 3073, 3074):
            q = component(
                job(kind, seed=seed, beta=300.0 if kind == "blend" else None),
                "planning",
            )
            delta = q["paths"]["CEM"][:, -1] - q["goals"]
            ax.scatter(delta[:, 0], delta[:, 1], s=15, color=COLORS[kind], alpha=0.4)
        ax.scatter([0], [0], marker="*", s=140, color="black")
        ax.add_patch(plt.Circle((0, 0), 1.5, color="black", fill=False, lw=0.8))
        ax.set(
            title=LABELS[kind],
            xlim=(-28, 28),
            ylim=(-28, 28),
            xlabel="Horizontal offset (px)",
            aspect="equal",
        )
    axs[0].set_ylabel("Vertical offset (px)")
    export(fig, folder / "trial_endpoints")
    fig, axs = plt.subplots(3, 6, figsize=(10, 5))
    for row, (label, images) in enumerate(saved["reconstruction"]):
        for col in range(6):
            axs[row, col].imshow(images[col], cmap="gray", vmin=-0.1, vmax=1.0)
            axs[row, col].set_xticks([])
            axs[row, col].set_yticks([])
        axs[row, 0].set_ylabel(label)
    export(fig, folder / "reconstruction_strip")
    for rho, condition in ((1.0, "frozen"), (0.0, "flicker")):
        panels = saved["variants"][condition]
        fig, axs = plt.subplots(4, 6, figsize=(10, 6))
        for row, (label, images) in enumerate(panels):
            for col in range(6):
                axs[row, col].imshow(images[col], cmap="gray", vmin=-0.1, vmax=1.0)
                axs[row, col].set_xticks([])
                axs[row, col].set_yticks([])
            axs[row, 0].set_ylabel(label, fontsize=8)
        export(fig, folder / f"variant_reconstructions_{condition}")
    backgrounds = saved["backgrounds"]
    frames = []
    for t in range(17):
        fig, axs = plt.subplots(1, 3, figsize=(7, 2.8))
        for ax, b, label in zip(axs, backgrounds, ["Frozen", "Slow drift", "Flicker"]):
            image = b[t]
            ax.imshow(image, cmap="gray", vmin=-0.25, vmax=1.0)
            ax.set_title(label)
            ax.axis("off")
        fig.suptitle("Same dot and moves · noise contrast ×4")
        fig.tight_layout()
        frames.append(frame(fig))
    gif(frames, folder / "backgrounds.gif")
    del backgrounds
    left = component(job("plain"), "planning")
    right = component(job("blend", beta=300.0), "planning")
    assert torch.equal(left["trajectory_ids"], right["trajectory_ids"])
    selected = saved["hero_selection"]["index"]
    frames = []
    for t in range(21):
        fig, axs = plt.subplots(1, 2, figsize=(6, 3.3))
        for i, (ax, q, kind) in enumerate(zip(axs, (left, right), ("plain", "blend"))):
            path = q["paths"]["CEM"][selected]
            image = saved["hero_images"][i, t]
            ax.imshow(image, cmap="gray", vmin=-0.1, vmax=1.0)
            ax.plot(path[: t + 1, 0], path[: t + 1, 1], color=COLORS[kind], lw=1.5)
            ax.scatter(
                path[0, 0],
                path[0, 1],
                marker="s",
                s=40,
                facecolors="none",
                edgecolors="cyan",
                label="Start",
            )
            goal = q["goals"][selected]
            ax.scatter(goal[0], goal[1], marker="*", s=90, color="gold", label="Target")
            if t == 20:
                ax.scatter(
                    path[-1, 0], path[-1, 1], marker="x", s=55, color="red", label="End"
                )
            ax.set_title(LABELS[kind])
            ax.axis("off")
        axs[0].legend(fontsize=7, loc="lower right")
        fig.suptitle(f"Frozen · typical paired trial · step {t}/20")
        fig.tight_layout()
        frames.append(frame(fig))
    gif(frames, folder / "hero.gif")
    save(folder / "hero_selection.json", saved["hero_selection"])
    return sorted(
        (str(x) for x in folder.iterdir() if x.suffix in (".png", ".svg", ".gif"))
    )
