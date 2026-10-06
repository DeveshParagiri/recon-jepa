import math

import torch
from data import IMAGE_SCALE, simulator


def bound_actions(actions):
    return actions * (4 / actions.norm(dim=-1, keepdim=True).clamp_min(4))


@torch.inference_mode()
def choose_actions(experiment, history, previous, goal, generator, config):
    device = experiment.device
    n = len(history)
    horizon, population = (config["horizon"], config["population"])
    mean = torch.zeros(n, horizon, 2, device=device)
    std = torch.full_like(mean, 2.0)
    pixel = experiment.config["objective"] == "pixel"
    for _ in range(config["iterations"]):
        noise = torch.randn(n, population, horizon, 2, generator=generator).to(device)
        candidates = bound_actions(mean[:, None] + std[:, None] * noise)
        candidates[:, 0] = bound_actions(mean)
        normalized = (candidates - experiment.action_mean) / experiment.action_std
        z = (
            history[:, None]
            .expand(-1, population, -1, -1)
            .reshape(n * population, 3, -1)
        )
        past = previous[:, None].expand(-1, population, -1, -1).reshape(-1, 2, 2)
        for future in range(horizon):
            action = normalized[:, :, future].reshape(-1, 1, 2)
            context = torch.cat((past, action), dim=1)
            predicted = experiment.model.predict(
                z, experiment.model.action_encoder(context)
            )[:, -1:]
            if pixel:
                predicted_pixels = experiment.decoder(predicted).reshape(
                    -1, 1, 1, 28, 28
                )
                if future + 1 < horizon:
                    predicted = experiment.encode(predicted_pixels / IMAGE_SCALE)
            z = torch.cat((z[:, 1:], predicted), dim=1)
            past = context[:, 1:]
        terminal = (
            predicted_pixels.flatten(1).reshape(n, population, -1)
            if pixel
            else z[:, -1].reshape(n, population, -1)
        )
        cost = (terminal - goal[:, None]).square().mean(-1)
        if not torch.isfinite(cost).all():
            raise FloatingPointError("Non-finite planning cost")
        ids = cost.topk(config["elites"], largest=False, dim=1).indices
        elite = candidates.gather(1, ids[:, :, None, None].expand(-1, -1, horizon, 2))
        mean, std = (elite.mean(1), elite.std(1).clamp_min(0.1))
    return candidates[torch.arange(n, device=device), cost.argmin(dim=1), 0].cpu()


def wilson(successes, total):
    z, p = (1.96, successes / total)
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = (
        z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    )
    return [center - radius, center + radius]


def plan(experiment, observations):
    experiment.eval()
    config = experiment.config["planning"]
    distances = (observations.positions[:, 16] - observations.positions[:, 2]).norm(
        dim=-1
    )
    eligible = torch.where(distances > 4)[0]
    ids = eligible[
        torch.randperm(
            len(eligible), generator=torch.Generator().manual_seed(config["seed"])
        )[: config["trials"]]
    ]
    if len(ids) != config["trials"]:
        raise ValueError(
            f"Expected {config['trials']} eligible goals; found {len(ids)}. Stop rather than silently reducing the test."
        )
    env = simulator()
    results, paths = ({}, {})
    for policy in ("CEM", "Random", "True-position control"):
        successes, final_distances, all_paths = ([], [], [])
        generator = torch.Generator().manual_seed(config["seed"])
        for group in ids.split(config["batch_size"]):
            current = observations.positions[group, 2].clone()
            goals = observations.positions[group, 16]
            reached = torch.zeros(len(group), dtype=torch.bool)
            trajectory = [current.clone()]
            if policy == "CEM":
                with torch.inference_mode():
                    history = experiment.encode(observations.images[group, :3])
                    goal = (
                        (observations.images[group, 16] * IMAGE_SCALE)
                        .flatten(1)
                        .to(experiment.device)
                        if experiment.config["objective"] == "pixel"
                        else experiment.encode(observations.images[group, 16:17])[:, 0]
                    )
                    previous = experiment.normalize_actions(
                        observations.actions[group, :2]
                    )
            for step in range(config["budget"]):
                if policy == "CEM":
                    action = choose_actions(
                        experiment, history, previous, goal, generator, config
                    )
                elif policy == "Random":
                    angle = 2 * math.pi * torch.rand(len(group), generator=generator)
                    length = 4 * torch.rand(len(group), generator=generator)
                    action = (
                        torch.stack((angle.cos(), angle.sin()), -1) * length[:, None]
                    )
                else:
                    action = bound_actions(goals - current)
                action = action.clone()
                action[reached] = 0
                sample = env.generate_transitions(current, action[:, None])
                current = sample.locations[:, -1, 0]
                reached |= (current - goals).norm(dim=-1) <= config["success_radius"]
                trajectory.append(current.clone())
                if policy == "CEM":
                    with torch.inference_mode():
                        observed = observations.observe(
                            sample.states[:, -1:], group, step + 3
                        )
                        history = torch.cat(
                            (history[:, 1:], experiment.encode(observed)), dim=1
                        )
                        normalized = experiment.normalize_actions(action)
                        previous = torch.cat(
                            (previous[:, 1:], normalized[:, None]), dim=1
                        )
                if reached.all():
                    break
            successes.extend(reached.tolist())
            final_distances.extend((current - goals).norm(dim=-1).tolist())
            while len(trajectory) < config["budget"] + 1:
                trajectory.append(current.clone())
            all_paths.append(torch.stack(trajectory, dim=1))
        count = sum(successes)
        results[policy] = dict(
            successes=count,
            trials=len(ids),
            success_rate=count / len(ids),
            wilson_95=wilson(count, len(ids)),
            mean_final_distance_px=sum(final_distances) / len(ids),
        )
        paths[policy] = torch.cat(all_paths)
    return dict(
        results=results,
        paths=paths,
        trajectory_ids=ids,
        goals=observations.positions[ids, 16],
        config=config,
    )


@torch.inference_mode()
def goal_background_check(experiment, observations, ids):
    config = experiment.config["planning"]
    donors = ids.roll(-1)
    paired = {name: [] for name in ("original", "swapped")}
    for name in paired:
        generator = torch.Generator().manual_seed(config["seed"])
        for offset in range(0, len(ids), config["batch_size"]):
            group = ids[offset : offset + config["batch_size"]]
            other = donors[offset : offset + config["batch_size"]]
            history = experiment.encode(observations.images[group, :3])
            previous = experiment.normalize_actions(observations.actions[group, :2])
            goal_image = (
                observations.images[group, 16:17]
                if name == "original"
                else observations.clean["images"][group, 16:17]
                + observations.strength
                / IMAGE_SCALE
                * observations.fields[other, 16:17]
            )
            goal = (
                (goal_image * IMAGE_SCALE).flatten(1).to(experiment.device)
                if experiment.config["objective"] == "pixel"
                else experiment.encode(goal_image)[:, 0]
            )
            paired[name].append(
                choose_actions(experiment, history, previous, goal, generator, config)
            )
    return dict(
        actions={k: torch.cat(v) for k, v in paired.items()},
        trajectory_ids=ids,
        donor_ids=donors,
        goals=observations.positions[ids, 16],
        threshold=1e-05,
        comparison="same dot target/start; only goal background replaced; identical CEM random candidates",
    )
