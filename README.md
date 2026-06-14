# HPE-Li DSK-Only 3D Ablation

Controlled 3D ablation of the HPE-Li++ pipeline on MMFi.

This repository keeps the training and model configuration used by
`HPE-Li-3D`, but removes every Channel Transformer module from the DSK
backbone. Its purpose is to measure the contribution of Transformer refinement
under otherwise matching experimental conditions.

## Ablation Contract

The following components are intentionally identical to `HPE-Li-3D`:

- Input: WiFi-CSI `(B, 3, 114, 10)`.
- Output: 3D pose `(B, 17, 3)`.
- Base channels: `128`, then `256`.
- DSK branches: `3`.
- Grouped convolution groups: `32`.
- SK reduction ratio: `4`.
- Regression head: `3584 -> 32 -> 64 -> 51`.
- MMFi dataset loader and preprocessing.
- Protocol 1 random split with ratio `0.8`.
- Sequence-level 50/50 validation/test partition.
- Training loss, optimizer defaults, checkpoint selection, and metrics.

The intended architecture difference is:

```text
HPE-Li++:
  CwSKA + FwSKA -> concat -> BatchNorm -> ChannelTransformer -> width pool

DSK-only ablation:
  CwSKA + FwSKA -> concat -> BatchNorm -> width pool
```

Verified default parameter counts:

| Model | Parameters | Transformer parameters |
|---|---:|---:|
| HPE-Li++ `DSKNetTransMMFI3D` | 2,056,851 | 1,663,488 |
| DSK-only `DSKNetMMFI3D` | 393,363 | 0 |

The parameter difference is exactly the Transformer parameter count. All 120
non-Transformer state-dict keys match between the two default models.

This is a paper-inspired HPE-Li DSK-only baseline with HPE-Li++ dimensions.
It is not a bit-for-bit reproduction of the original ECCV 2024
hyperparameters, which used a different model width and 2D output.

## Project Layout

```text
config/mmfi/
  config_p1s1.yaml            Default Protocol 1 Setting 1 experiment
  config.yaml                 Generic Protocol 3 preset
docs/
  model_logic_vi.md           Vietnamese architecture and ablation notes
feeder/
  mmfi.py                     MMFi dataset and dataloader
  splits.py                   Sequence-level validation/test split
model/
  dsknet3d.py                 DSKNetMMFI3D without Transformer
  utils/regression.py         Shared regression head
tools/
  evaluate.py                 Checkpoint evaluation
utils/
  eval_3d.py                  Shared 3D metrics
tests/
  test_dsk_only_pipeline.py   Shape, checkpoint, and ablation tests
train.py                      Training entry point
```

## Training

Set the MMFi dataset root:

```powershell
$env:MMFI_DATASET_ROOT = "D:\path\to\mmfi\dataset"
```

Run the default experiment:

```powershell
python train.py
```

Run a short smoke test:

```powershell
python train.py --epochs 1 --max-train-batches 1 --eval-max-batches 1
```

Results are written to `results/dsk_only` by default.

## Evaluation

```powershell
python tools/evaluate.py `
  --checkpoint results/dsk_only/<run>/checkpoints/best.pt `
  --eval-split test
```

## Metrics

- MPJPE and PA-MPJPE in millimetres.
- PCK@50mm and PCK@100mm.
- Body-scale `g_PCK@10` through `g_PCK@50`.
- Per-joint and collapse-diagnostic metrics.

These definitions are copied from `HPE-Li-3D` to keep evaluation parity.

## Tests

```powershell
python -m unittest discover -s tests -v
```
