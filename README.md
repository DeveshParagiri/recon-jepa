# recon-jepa

Experiments on whether image reconstruction helps a world model track a moving dot against predictable backgrounds. The comparison covers LeWM, LeWM with reconstruction, next-image prediction, and MotionJEPA.

## Setup

Use Python 3.12. Run these commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

`main.py` generates the data, trains and evaluates the models, then produces tables and figures. Settings are in `config.yaml`; outputs go to `outputs/moving-dot/`. MotionJEPA's loss implementation is downloaded on first use.

The default device is CUDA when available, otherwise CPU. The reference experiments used NVIDIA L40S GPUs, float32 training, and no TF32, mixed precision, or compilation.

To run a single stage:

```bash
python main.py --stage pilot
python main.py --stage validation
python main.py --stage test
python main.py --stage report
```

The pilot trains and validates one reconstruction model with frozen backgrounds. Validation runs the comparisons and selects a reconstruction variant. Test evaluates the frozen models on held-out data. Report generates the tables and figures.

Repeat a command to resume from its checkpoint or reuse completed evaluations. Checkpoints include optimizer, scheduler, sampler, and random states. Use `--study another-name` for a separate experiment. Resume with the same code and hardware, and run only one instance of each study at a time.

## Modal

Install the client and authenticate with your Modal account:

```bash
pip install modal==1.6.1 PyYAML==6.0.3
modal token new
python main.py --backend modal --stage pilot
```

After the pilot finishes, run the full study. It reuses the pilot model:

```bash
python main.py --backend modal
```

Modal installs the remaining dependencies and runs up to five L40S workers. Set the app name, Volume name, and concurrency in `config.yaml`. GPU usage is billed to your account.

Closing the terminal leaves the run active. Use Modal's dashboard to read logs or stop it. Once it has stopped, repeat the command to resume. Add `--stage` to run any of the stages above on Modal.

Download results with the default Volume and study names:

```bash
modal volume get recon-jepa /moving-dot outputs/moving-dot
```

With the full dependencies installed locally, run `python main.py --stage report` to regenerate tables and figures.

## Experiments

| Model | Training loss |
| --- | --- |
| LeWM | Latent prediction + 0.09 × SIGReg |
| LeWM + reconstruction | LeWM + 300 × current-image MSE |
| Pixel model | Next-image MSE |
| MotionJEPA | Upstream static, difference, and inverse losses, weighted 0.25, 2, and 0.5 |

The main comparison uses background correlations of `1`, `0.99`, `0.9`, `0.5`, and `0`, with training seeds `3072`, `3073`, and `3074`. Correlation 1 freezes the background; 0 redraws it each frame. Noise strength is 0.03 relative to the theoretical dot peak. Paired models share trajectories, backgrounds, and goals.

Models use a six-layer, width-96 encoder, a two-layer predictor, and three frames of context. Training uses 5,000 AdamW updates, batch size 128, peak learning rate 5e-5, 50 warmup steps, cosine decay, and weight decay 0.001. MotionJEPA has an independent difference encoder; planning uses its frame representation.

Additional runs cover clean backgrounds, reconstruction weights of `0.1`, `1`, `30`, and `300`, and blurred, dot-weighted, and decaying reconstruction losses. Validation determines whether to extend one of these variants across all correlations and seeds. This gives 72 models, or 85 with the extension. Dot-weighted reconstruction uses clean-dot information during training and is reported separately from the main comparison.

Each trajectory has 17 frames at 28 × 28 pixels:

| Split | Trajectories | Trajectory seed | Noise seed | Planning goals |
| --- | ---: | ---: | ---: | ---: |
| Train | 20,000 | 0 | 10000 | — |
| Validation | 2,000 | 1 | 10001 | 100 |
| Test | 1,024 | 32452843 | 32452867 | 500 |

CEM uses 128 candidates, 16 elites, five refinements, a five-action horizon, and a 20-action budget. Success means reaching within 1.5 pixels of the goal. Evaluation also measures position readouts fitted on training data, prediction and shuffled-action errors, background retrieval, and sensitivity to the goal's background. Random actions and true-position planning provide controls.

Models and readouts are frozen before test data is generated. MotionJEPA must improve frozen-background validation planning over LeWM in every seed and by at least 10 percentage points on average; otherwise the run stops for method inspection.

The main test criterion requires reconstruction gains of at least 20 percentage points at correlations 1 and 0.99, and an absolute difference within 10 points at correlation 0, in every seed.

## Sources

Model components are adapted from [LeWM](https://github.com/lucas-maes/le-wm); the simulator comes from [JEPA_SSL_NeurIPS_2022](https://github.com/vladisai/JEPA_SSL_NeurIPS_2022). Their MIT notices are retained in `vendor/`. The MotionJEPA loss uses the [upstream implementation](https://github.com/mkarmann/motion-jepa).
