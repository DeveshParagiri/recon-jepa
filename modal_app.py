import os
import sys
from pathlib import Path

import modal
import yaml

BASE = Path(__file__).resolve().parent if modal.is_local() else Path("/opt/study")
if not modal.is_local():
    sys.path.insert(0, str(BASE))
CONFIG = (
    Path(os.environ.get("STUDY_CONFIG", BASE / "config.yaml"))
    if modal.is_local()
    else BASE / "config.yaml"
)
settings = yaml.safe_load(CONFIG.read_text())["modal"]
app = modal.App(settings["app"])
volume = modal.Volume.from_name(settings["volume"], create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(BASE / "requirements.txt")
    .env(
        {
            "PYTHONPATH": "/opt/study",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "MPLBACKEND": "Agg",
        }
    )
    .add_local_dir(
        BASE / "vendor", "/opt/study/vendor", ignore=["**/__pycache__/**", "**/*.pyc"]
    )
)
for path in BASE.glob("*.py"):
    image = image.add_local_file(path, "/opt/study/" + path.name)
image = image.add_local_file(CONFIG, "/opt/study/config.yaml")


@app.function(
    image=image,
    volumes={"/results": volume},
    gpu="L40S",
    cpu=2,
    memory=16384,
    max_containers=5,
    timeout=3600,
    retries=0,
)
def worker(config, task):
    from main import execute

    volume.reload()
    execute(Path("/results") / config["study"], config, task, "cuda", volume.commit)


@app.function(
    image=image,
    volumes={"/results": volume},
    cpu=2,
    memory=16384,
    timeout=86400,
    retries=0,
    max_containers=1,
)
def run_study(config, stage):
    from main import execute, run

    root = Path("/results") / config["study"]
    volume.reload()

    def dispatch(tasks):
        if tasks[0]["phase"] not in ("validation", "test"):
            execute(root, config, tasks[0], "cpu", volume.commit)
            return True
        limit = config["modal"]["concurrency"]
        for start in range(0, len(tasks), limit):
            calls = [
                worker.spawn(config, task) for task in tasks[start : start + limit]
            ]
            errors = []
            for call in calls:
                try:
                    call.get()
                except Exception as error:
                    errors.append(error)
            volume.reload()
            if errors:
                raise RuntimeError(f"Worker failed: {errors[0]}") from errors[0]
        return True

    run(root, config, stage, dispatch, volume.commit, volume.reload)


@app.local_entrypoint()
def main(stage: str = "all", study: str = "moving-dot"):
    from main import load_config

    config = load_config(CONFIG)
    config["study"] = study
    run_study.remote(config, stage)
