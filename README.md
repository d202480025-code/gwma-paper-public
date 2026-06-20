# GWMA

Minimal public code for reproducing the GWMA paper experiments. Generated
datasets, checkpoints, and result folders are not included.

## Install

Run from WSL:

```bash
cd gwma-paper-public
export PATH="$HOME/.local/bin:$PATH"
uv sync --extra gw
```

## GWMA Configuration

The uploaded version corresponds to the current best GWMA recipe used for the
paper revision: stronger MAE pretraining with `mask_ratio=0.85` for 50 epochs,
followed by 40 epochs of supervised fine-tuning.

The main architecture parameters are in `configs/model/gwma.yaml`:

- signal length: 4096
- frame length / hop length: 64 / 32
- embedding dimension: 768
- depth / attention heads: 24 / 12
- MLP ratio: 4.0
- model default mask ratio: 0.75
- best pre-training mask ratio: 0.85
- embedding / decoder: `conv` / `conv_transpose`
- checkpointing: enabled
- default inference veto: enabled

Training parameters are in:

- `configs/experiment/pretrain.yaml`
- `configs/experiment/finetune.yaml`
- `configs/experiment/ablation_no_pretrain.yaml`
- `configs/experiment/baseline_unet.yaml`
- `configs/experiment/baseline_bilstm.yaml`

The overall reproduction plan is `configs/experiment/paper_main.yaml`.

Reference best-GWMA metrics from the summary folder:

| Dataset | Metric | Value |
|---|---|---:|
| Gaussian mixed-SNR | mean overlap | 0.9719 |
| Normal glitch mixed-SNR | mean overlap | 0.9365 |
| Standard pure glitch safety | mean output/input ratio | 0.0056 |

## Default Veto

GWMA includes the conservative inference-time veto as part of the model
configuration, not as an external plugin. It is enabled by default for
`eval()` inference when `mask_ratio=0.0`, and it is disabled during training.

Default thresholds in `configs/model/gwma.yaml`:

- `veto_input_energy_floor: 100.0`
- `veto_output_energy_floor: 0.0`
- `veto_output_input_ratio_floor: 0.00375`
- `veto_apply_during_training: false`

With the best checkpoint, this setting leaves Gaussian unchanged, changes the
normal glitch mean overlap by about `0.0013`, and vetoes the high-overlap
pure-glitch stress set in the internal safety scan.

## Reproduce

```bash
uv run python scripts/reproduce_paper.py --stage data
uv run python scripts/reproduce_paper.py --stage train
uv run python scripts/reproduce_paper.py --stage evaluate
uv run python scripts/reproduce_paper.py --stage tables
```

## Outputs

- `data/`: generated HDF5 datasets.
- `results/pretrain_stronger_mask085_e50/`, `results/finetune_stronger_pretrain/`,
  `results/ablation_no_pretrain_conv_hilbert/`,
  `results/baseline_unet_hilbert/`,
  `results/baseline_bilstm_hilbert/`: training logs and checkpoints.
- `results/paper/<model>_<dataset>/`: per-sample evaluation CSVs and
  summary JSON files.
- `results/paper/tables/`: main reproduction CSV/Markdown tables.
