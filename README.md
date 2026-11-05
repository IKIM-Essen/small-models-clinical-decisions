# Small fine-tuned models versus frontier models for clinical decisions

Code and aggregate results from the Institute for Artificial Intelligence in Medicine (IKIM), University Hospital Essen.

We fine-tuned two 4-billion-parameter models, Kev-4B (a System One decision model) and Qwen3.5-4B with LoRA, on
3,000 examples per task and compared them with two prompted open-weight frontier models on six clinical
classification tasks: three on public PhysioNet data and three on internal data of a German university hospital.
The primary endpoint was automation at a 5% error budget, the share of decisions a model can make on its own while
deferring the rest to a clinician.

| Task | Data | Question |
|---|---|---|
| P1 radiology findings | MIMIC-CXR | Which of 14 findings does a chest radiograph report describe? |
| P2 clinical inference | MedNLI | Does a statement follow from a clinical note sentence, contradict it, or neither? |
| P3 triage acuity | MIMIC-IV-ED | Emergency Severity Index level recorded at triage |
| I1 length of stay | internal | Length of an inpatient stay in four bands, from admission information |
| I2 discharge-letter ICD-10 chapter | internal | ICD-10 chapter of the main diagnosis from a discharge letter |
| I3 pathology malignancy | internal | Malignant tumor, and if so metastatic, from a pathology report |

## Reproduce the results file

`runs/results/metrics.json` collects the hypothesis tests, latency, sensitivity and subgroup results. It is computed
from the aggregate run outputs in `runs/`. Rebuilding it takes seconds, needs only
[uv](https://docs.astral.sh/uv/), and uses no GPU and no patient data:

```bash
make reproduce   # runs/ -> runs/results/metrics.json, then checks it matches the committed copy
```

## Repository layout

| Path | Contents |
|---|---|
| `src/kevmed/` | Task builders (one module per task), split logic, schema, token counting, extraction queries for the internal tasks |
| `scripts/data/` | Building and checking splits, PhysioNet file layout, model downloads, internal extraction |
| `scripts/qwen/` | LoRA supervised fine-tuning and scoring of Qwen3.5-4B |
| `scripts/frontier/` | Frontier-model scoring through an OpenAI-compatible gateway, label-definition prompts |
| `scripts/classical/` | TF-IDF and gradient-boosted tree baselines |
| `scripts/tier2c/`, `scripts/laya/` | Servers for the zero-shot System One models scored on development data |
| `scripts/analysis/` | Freezing configurations, hypothesis tests, bootstrap, latency, sensitivity and subgroup analyses |
| `scripts/repro/` | Environment and split checks, `reproduce.sh` |
| `scripts/review/` | Label spot-check and outcome-audit tools |
| `configs/` | Frozen configuration of every test run (`frozen.json`), model and code pins |
| `runs/` | Aggregate results only: metrics, hypothesis tests, curves, latency, subgroup and sensitivity summaries |
| `splits/` | Build provenance, token statistics and SHA-256 checksums of the frozen splits (no records) |
| `tests/` | Unit tests on synthetic fixtures |
| `third_party/` | Kev and Laya as git submodules, pinned to the commits used in the study |

## Re-running the experiments

Re-running training and scoring needs credentialed data access and GPUs:

1. `git submodule update --init` and `uv sync` (Python 3.13, Linux, CUDA 12).
2. Public tasks: credentialed PhysioNet access to MIMIC-CXR, MIMIC-CXR-JPG, MedNLI, MIMIC-IV and MIMIC-IV-ED.
   Arrange the downloaded files with `scripts/data/physionet_layout.py`, build each task with
   `scripts/data/build_task.py`, and check the result against `splits/CHECKSUMS.sha256` with
   `scripts/repro/verify_splits.py`.
3. Internal tasks: the data cannot leave University Hospital Essen. The task builders in `src/kevmed/tasks/` and the
   queries in `src/kevmed/fmx/` document exactly how records and labels were derived.
4. Fine-tuning used one A100 80 GB GPU per run. `configs/frozen.json` records, for every test run, the commit,
   seed, context length, development-set temperature and the SHA-256 of the test file it was scored on.
5. Frontier models were reached through an OpenAI-compatible LiteLLM gateway. `scripts/frontier/serve.py` reads
   `LLM_GATEWAY_URL`, `KEV_LITELLM_KEY` and, optionally, `LLM_EXPECTED_API_BASE` from the environment.

Run `make test` for the unit tests.

## Data

The PhysioNet datasets are available to credentialed users under their data use agreements and are not
redistributed here. The internal data contain patient information and cannot be shared; all results on them are
reported in aggregate. No file in this repository contains a patient record.

## Licence

The code is released under the [Apache License 2.0](LICENSE). Kev and Laya are separate projects under their own
licences.

## Security

`uv.lock` takes published security fixes where they leave the software environment intact. PyTorch stays at 2.8.0:
Kev at the commit used here requires torch < 2.9, and the prebuilt causal-conv1d kernels exist only up to
torch 2.10. Of the open PyTorch advisories, one concerns this code: loading an untrusted checkpoint with `torch.load`
can execute code (GHSA-63cw-57p8-fm3p), so load only the published Kev weights and checkpoints you trained yourself.
The other three concern `torch.jit.script`, `torch.lstm_cell` and `unpack_sequence`, which this code does not call.

## Disclaimer

This is research code. It is not a medical device and must not be used for clinical decisions.
