# AGENTS.md

## Cursor Cloud specific instructions

This repo is the **VLA Interpretability Handoff**: a Python CLI/script research toolkit (no servers, DB, or frontend). "Running it" means executing the numbered analysis scripts in `scripts/` driven by `configs/demo.yaml`. See `README.md` for the full per-demo command reference.

### Environment

- Base scientific dependencies (`requirements.txt`) are installed with `pip install --user` (the VM has Python 3.12; `conda` and `python3-venv` are not available, so the README's conda flow does not apply here). The base stack runs fine on Python 3.12.
- The base deps are enough to run **Demo 1 in mock/synthetic mode end-to-end on CPU** (no GPU, no network, no model checkpoints).
- The "real" path (`--model pi05`, `--env libero`/`libero_dataset`, all of Demo 2 & Demo 3) additionally needs a CUDA GPU + `lerobot[pi,libero]` (large git dependency) + LIBERO/MuJoCo with EGL rendering + PI0/PI0.5 checkpoints. None of these are present in the cloud VM, so those demos cannot be run here without provisioning them.

### Running the lightweight Demo 1 pipeline (the CPU smoke/hello-world path)

`configs/demo.yaml` is preconfigured for the real `libero_dataset` + `pi05` path and **lacks the keys the synthetic env needs** (`env.min_pixel_distance`, `env.background_randomization`). To run the mock path, use a config that sets `env.name: synthetic` (with those two keys) and `model.name: mock`. A ready copy is kept at `/tmp/demo_synthetic.yaml` during setup; recreate it from `configs/demo.yaml` if missing.

Non-obvious gotcha: `scripts/03_train_layerwise_probe.py --target all` fails on the synthetic+mock path because `action_chunk`, `gt_action`, and `gt_action_chunk` targets are never produced (mock returns no action chunk; synthetic states have no ground-truth labels). Train the working targets individually (`offset`, `target_position`, `gripper_position`, `action`); `scripts/05_plot_results.py --target all` then builds the combined R² figure and simply skips the missing targets.

Full mock run:

```bash
python3 scripts/01_collect_states.py --config /tmp/demo_synthetic.yaml
python3 scripts/02_extract_activations.py --config /tmp/demo_synthetic.yaml --model mock
for t in offset target_position gripper_position action; do
  python3 scripts/03_train_layerwise_probe.py --config /tmp/demo_synthetic.yaml --target $t
done
python3 scripts/05_plot_results.py --config /tmp/demo_synthetic.yaml --target all
```

Outputs land under `outputs/` (gitignored): `outputs/activations/activations.npz`, `outputs/probes/*.csv`, `outputs/figures/layerwise_probe_targets_r2_comparison.png`.

### Lint / test / build

There is no lint config, no automated test suite, and no build step in this repo. The canonical "does the code import/parse" check is `py_compile` (from the README):

```bash
python3 -m py_compile scripts/03_train_layerwise_probe.py scripts/05_plot_results.py scripts/18_plot_pi0_ablation_heatmaps.py
```
