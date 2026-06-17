# DSKNet 3D No-Transformer Baseline

This repository implements the DSKConv logic from
`HPE-Li-ECCV2024/model/sknet_trans_mmfi.py` and ports it to 3D pose output.

The source module names the block `SKConv`, but its forward path is the DSKConv
logic used by DSKNet:

```text
multi-dilation conv branches
  -> stack branches
  -> channel-wise selective kernel attention
  -> frequency-wise selective kernel attention
  -> concat channel/frequency outputs along width
  -> BatchNorm
  -> optional ChannelTransformer in forked source
  -> AvgPool2d(1, 2)
```

In this repo, the ChannelTransformer is intentionally removed. Everything else
in the DSKConv path is kept.

## Architecture

```text
Input (B, 3, 114, 10)
  -> DSKUnit 1: Conv1x1, pool, DSKConv, Conv1x1
  -> (B, 128, 57, 5)
  -> BatchNorm
  -> DSKUnit 2: Conv1x1, pool, DSKConv, Conv1x1
  -> (B, 256, 28, 2)
  -> AvgPool2d(2, 2)
  -> (B, 256, 14, 1)
  -> Regression 3584 -> 32 -> 64 -> 51
  -> Output (B, 17, 3)
```

Default model config:

```python
{
    "num_lay": 128,
    "hidden_reg": 32,
    "sk_m": 3,
    "sk_g": 32,
    "sk_r": 4,
    "sk_l": 32,
}
```

Default parameter count: `393,363`.

## DSKConv

For input `x: (B, C, H, W)`, DSKConv does:

```python
feats = torch.stack([conv(x) for conv in self.convs], dim=1)
```

Shape:

```text
feats: (B, M, C, H, W)
```

Channel-wise selection:

```text
sum over M
  -> global average pool over H,W
  -> bottleneck Conv2d
  -> branch-specific Conv2d
  -> softmax over M
  -> weighted sum
```

Frequency-wise selection:

```text
sum over C
  -> average pool over W only
  -> softmax over M
  -> weighted sum
```

Fusion:

```text
feats_channel:   (B, C, H, W)
feats_frequency: (B, C, H, W)
concat width:    (B, C, H, 2W)
BatchNorm
AvgPool2d(1, 2): (B, C, H, W)
```

No `ChannelTransformer` is created or called in this repo.

## Experiment Profile

- MMFi modality: WiFi CSI
- Protocol: P1
- Setting: S1 random split
- Train/eval ratio: `0.8/0.2`, split seed `0`
- Validation/test split: by sequence, seed `41`
- Train/val/test batch size: `16/8/8`
- Pose target: train-set XYZ z-score
- Loss: MSE
- Optimizer: Adam, learning rate `0.001`, weight decay `0`
- Scheduler: none
- Gradient clipping: `1.0`
- Maximum epochs: `60`
- Early stopping: patience `15`, minimum delta `0.2 mm`
- Best checkpoint: minimum validation MPJPE

## Training

```powershell
$env:MMFI_DATASET_ROOT = "D:\path\to\mmfi\dataset"

python train.py --device cuda --seed 0 `
  --run-name dsknet_3d_no_transformer_p1s1_seed0

python train.py --device cuda --seed 1 `
  --run-name dsknet_3d_no_transformer_p1s1_seed1
```

## Evaluation

```powershell
python tools/evaluate.py `
  --checkpoint results/phase_c/<run>/checkpoints/best.pt `
  --dataset-root $env:MMFI_DATASET_ROOT `
  --eval-split test
```

## Tests

```powershell
python -m unittest discover -s tests -v
```
