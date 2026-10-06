import hashlib
import sys
import urllib.request
from pathlib import Path

import torch
from data import IMAGE_SCALE
from torch import nn
from transformers import ViTConfig, ViTModel
from utils import seed_all
from vendor.lewm.jepa import JEPA
from vendor.lewm.module import MLP, ARPredictor, Embedder, SIGReg


def encoder(settings):
    cfg = ViTConfig(
        hidden_size=settings["width"],
        num_hidden_layers=settings["layers"],
        num_attention_heads=settings["encoder_heads"],
        intermediate_size=settings["encoder_mlp"],
        image_size=28,
        patch_size=settings["patch"],
    )
    result = ViTModel(cfg, add_pooling_layer=False, use_mask_token=False)
    result.config.interpolate_pos_encoding = True
    return result


def projector(settings):
    return MLP(
        settings["width"],
        settings["encoder_mlp"],
        settings["width"],
        norm_fn=nn.BatchNorm1d,
    )


def beta_at(job, step, total=5000):
    return (
        job["beta"] * (total - step) / (total - 1)
        if job.get("variant") == "decay"
        else job["beta"]
    )


def blur(x):
    axis = torch.arange(-3, 4, device=x.device, dtype=x.dtype)
    k = torch.exp(-axis.square() / 2)
    k = k / k.sum()
    kernel = (k[:, None] * k[None, :])[None, None]
    return torch.nn.functional.conv2d(
        torch.nn.functional.pad(x.reshape(-1, 1, 28, 28), (3, 3, 3, 3), mode="reflect"),
        kernel,
    ).reshape_as(x)


class Experiment(nn.Module):
    def __init__(
        self, protocol, job, normalization, device="cuda", official_cache=None
    ):
        super().__init__()
        self.protocol = protocol
        self.job = job
        self.device = torch.device(device)
        self.config = {"planning": protocol["planning"], "objective": job["objective"]}
        settings = protocol["model"]
        width = settings["width"]
        seed_all(job["seed"])
        base = JEPA(
            encoder(settings),
            ARPredictor(
                num_frames=3,
                input_dim=width,
                hidden_dim=width,
                output_dim=width,
                depth=settings["predictor_layers"],
                heads=settings["predictor_heads"],
                dim_head=settings["head_dim"],
                mlp_dim=settings["encoder_mlp"],
                dropout=settings["dropout"],
                emb_dropout=0.0,
            ),
            Embedder(input_dim=2, emb_dim=width),
            projector(settings),
            projector(settings),
        )
        self.sigreg = SIGReg(knots=17, num_proj=1024)
        self.decoder = None
        self.model = base
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(job["seed"] + 101)
            if job["objective"] == "motionjepa":
                Motion, Frame = load_motion(official_cache)

                class PlanningMotion(Motion):
                    def encode(self, info):
                        if "action" in info and info["pixels"].shape[1] > 1:
                            return super().encode(info)
                        return Frame.encode(self, info)

                self.model = PlanningMotion(
                    encoder=base.encoder,
                    predictor=base.predictor,
                    action_encoder=base.action_encoder,
                    projector=base.projector,
                    pred_proj=base.pred_proj,
                    diff_encoder=encoder(settings),
                    diff_projector=projector(settings),
                    embed_dim=width,
                    history_size=3,
                    num_preds=1,
                    sigreg=self.sigreg,
                    sigreg_weight=0.25,
                    diff_sigreg_weight=2.0,
                    diff_pred_weight=0.5,
                )
            elif job["objective"] in ("blend", "pixel"):
                self.decoder = nn.Sequential(
                    nn.Linear(width, settings["encoder_mlp"]),
                    nn.GELU(),
                    nn.Linear(settings["encoder_mlp"], 784),
                )
        self.register_buffer("action_mean", torch.tensor(normalization["mean"]))
        self.register_buffer("action_std", torch.tensor(normalization["std"]))
        self.register_buffer(
            "image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
        )
        self.to(device).float()

    def pixels(self, x):
        return (
            (x.to(self.device) * IMAGE_SCALE).repeat(1, 1, 3, 1, 1) - self.image_mean
        ) / self.image_std

    def encode(self, x):
        return self.model.encode({"pixels": self.pixels(x)})["emb"]

    def normalize_actions(self, a):
        return (a.to(self.device) - self.action_mean) / self.action_std

    def forward(self, raw, step=None, shuffle=False, return_predictions=False):
        if step is None:
            step = self.protocol["training"]["steps"]
        images, actions, clean = raw
        pixels = self.pixels(images)
        actions = self.normalize_actions(actions)
        if shuffle:
            actions = actions.roll(1, 0)
        target = (images * IMAGE_SCALE).to(self.device).flatten(2)
        kind = self.job["objective"]
        if kind == "motionjepa":
            emb, terms = self.model.train_step({"pixels": pixels, "action": actions})
            pred = None
            if return_predictions:
                output = self.model.encode({"pixels": pixels, "action": actions})
                pred = self.model.predict(
                    output["emb"][:, :3], output["act_emb"][:, :3]
                )
            zero = emb.new_zeros(())
            return dict(
                emb=emb,
                loss=terms["loss"],
                pred_loss=terms["pred"],
                sigreg_loss=terms["static_sigreg"],
                recon_loss=zero,
                diff_sigreg_loss=terms["diff_sigreg"],
                diff_pred_loss=terms["inv"],
                persistence_mse=(emb[:, 1:] - emb[:, :-1]).square().mean(),
                predicted=pred,
                target=emb[:, 1:] if return_predictions else None,
                persistent=emb[:, :-1] if return_predictions else None,
            )
        out = self.model.encode(
            {"pixels": pixels[:, :3] if kind == "pixel" else pixels, "action": actions}
        )
        emb = out["emb"]
        pred = self.model.predict(emb[:, :3], out["act_emb"][:, :3])
        zero = emb.new_zeros(())
        sig = rec = zero
        if kind == "pixel":
            pred = self.decoder(pred)
            tgt = target[:, 1:]
            persistent = target[:, :-1]
            loss = pred_loss = (pred - tgt).square().mean()
        else:
            tgt = emb[:, 1:]
            persistent = emb[:, :-1]
            pred_loss = (pred - tgt).square().mean()
            sig = self.sigreg(emb.transpose(0, 1))
            if kind == "blend":
                rec_target = (
                    blur(images.to(self.device) * IMAGE_SCALE).flatten(2)
                    if self.job.get("variant") == "blur"
                    else target
                )
                error = (self.decoder(emb) - rec_target).square()
                if self.job.get("variant") == "weighted":
                    assert clean is not None
                    error = error * (
                        0.05 + 0.95 * (clean.to(self.device) * IMAGE_SCALE).flatten(2)
                    )
                rec = error.mean()
            loss = (
                pred_loss
                + 0.09 * sig
                + beta_at(self.job, step, self.protocol["training"]["steps"]) * rec
            )
        return dict(
            emb=emb,
            loss=loss,
            pred_loss=pred_loss,
            sigreg_loss=sig,
            recon_loss=rec,
            diff_sigreg_loss=zero,
            diff_pred_loss=zero,
            persistence_mse=(tgt - persistent).square().mean(),
            predicted=pred if return_predictions else None,
            target=tgt if return_predictions else None,
            persistent=persistent if return_predictions else None,
        )


OFFICIAL_COMMIT = "b3c2fb374b7c69f169de15774f371290a8736f56"
HASHES = {
    "src/models/motionjepa/jepa.py": "4cb7f746ac3f84db6def779fb6651070f91d47bcd32e95efc9eec4b0d1965f25",
    "src/models/lewm/jepa.py": "cb8d08750880ed82794f5ca83004e3262f77c9c96abba8031545a9fb878207f8",
    "src/models/lewm/module.py": "452de90884bec44217fe74209ffae50d289ccaf1b9c3aff4f5f9e4a8631081b5",
}


def load_motion(cache):
    root = Path(cache) / "motionjepa"
    root.mkdir(parents=True, exist_ok=True)
    for relative, expected in HASHES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            content = urllib.request.urlopen(
                f"https://raw.githubusercontent.com/mkarmann/motion-jepa/{OFFICIAL_COMMIT}/{relative}",
                timeout=60,
            ).read()
            assert hashlib.sha256(content).hexdigest() == expected
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(content)
            temporary.replace(path)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        parent = path.parent
        while parent != root:
            if not (parent / "__init__.py").exists():
                (parent / "__init__.py").touch()
            parent = parent.parent
    sys.path.insert(0, str(root))
    from src.models.lewm.jepa import JEPA
    from src.models.motionjepa.jepa import MotionJEPA

    return (MotionJEPA, JEPA)
