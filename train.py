"""
train.py
Training entry point for the DSKNet 3D no-Transformer baseline on MMFi.

Typical usage
-------------
    # P1-S1 canonical run: normalized MSE, Adam, fixed 50 epochs
    python train.py

    # One-batch parity smoke test
    python train.py --epochs 1 --max-train-batches 1 --eval-max-batches 1

Environment variables
---------------------
    MMFI_DATASET_ROOT            Path to MMFi dataset root.
    PHASE_C_OUTPUT               Output directory for runs.
    PHASE_C_EPOCHS               Number of epochs.
    PHASE_C_LR                   Learning rate.
"""
import argparse
import copy
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feeder import make_dataloader, make_dataset
from feeder.splits import split_eval_dataset_by_sequence
from model.dsknet3d import (
    CHECKPOINT_FORMAT_VERSION,
    MODEL_NAME,
    DSKNetMMFI3D,
)
from utils.eval_3d import compute_3d_metrics


P1S1_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "mmfi" / "config_phase_c_demo_2_p1s1.yaml"
)
POSE_STD_EPS = 1e-6
GRAD_CLIP_NORM = 1.0


# ─── SECTION 1: CLI & Setup ───────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train the DSKNet 3D no-Transformer baseline on MMFi."
    )
    p.add_argument("--dataset-root",
                   default=os.getenv("MMFI_DATASET_ROOT",
                                     str(PROJECT_ROOT / "data" / "mmfi" / "dataset")),
                   help="MMFi dataset root directory.")
    # Output
    p.add_argument("--output-dir",
                   default=os.getenv("PHASE_C_OUTPUT",
                                     str(PROJECT_ROOT / "results" / "phase_c")),
                   help="Root directory for logs, metrics, and checkpoints.")
    p.add_argument("--run-name", default=None,
                   help="Optional run name (default: timestamp + split).")
    # Training
    p.add_argument("--epochs",       type=int,   default=int(os.getenv("PHASE_C_EPOCHS", "50")))
    p.add_argument("--lr",           type=float, default=float(os.getenv("PHASE_C_LR", "0.001")))
    p.add_argument("--seed",         type=int,   default=0)
    p.add_argument("--device",       default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--log-interval",   type=int,   default=100)
    # Debug / batch limits
    p.add_argument("--eval-max-batches",  type=int, default=None)
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--train-batch-size",  type=int, default=None)
    p.add_argument("--val-batch-size",    type=int, default=None)
    p.add_argument("--test-batch-size",   type=int, default=None)
    p.add_argument("--num-workers",       type=int, default=None)
    return p.parse_args()


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_arg)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── SECTION 2: Data ──────────────────────────────────────────────────────────

def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def validate_p1s1_config(config):
    expected = {
        "modality": "wifi-csi",
        "protocol": "protocol1",
        "data_unit": "frame",
        "split_to_use": "random_split",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"P1-S1 requires {key}={value!r}, got {config.get(key)!r}.")
    random_split = config.get("random_split", {})
    if random_split.get("ratio") != 0.8 or random_split.get("random_seed") != 0:
        raise ValueError("P1-S1 requires random_split ratio=0.8 and random_seed=0.")


def apply_loader_overrides(config, args):
    config = copy.deepcopy(config)
    for key in ("train_loader", "val_loader", "test_loader"):
        config[key] = dict(config[key])
    if args.train_batch_size is not None:
        config["train_loader"]["batch_size"] = args.train_batch_size
    if args.val_batch_size is not None:
        config["val_loader"]["batch_size"] = args.val_batch_size
    if args.test_batch_size is not None:
        config["test_loader"]["batch_size"] = args.test_batch_size
    if args.num_workers is not None:
        for key in ("train_loader", "val_loader", "test_loader"):
            config[key]["num_workers"] = args.num_workers
    return config


def make_run_dir(output_dir, run_name, split_to_use):
    if run_name is None:
        run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{split_to_use}"
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    return run_dir


def make_loaders(dataset_root, config, seed):
    train_ds, eval_ds = make_dataset(dataset_root, config)
    gen = torch.Generator().manual_seed(seed)

    train_loader = make_dataloader(train_ds, is_training=True,  generator=gen, **config["train_loader"])
    val_ds, test_ds, split_meta = split_eval_dataset_by_sequence(eval_ds, test_size=0.5, random_state=41)
    val_loader  = make_dataloader(val_ds,  is_training=False, generator=gen, **config["val_loader"])
    test_loader = make_dataloader(test_ds, is_training=False, generator=gen, **config["test_loader"])

    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds, split_meta


# ─── SECTION 3: Pose Normalisation ───────────────────────────────────────────

def compute_pose_normalization_stats(dataset, eps=1e-6):
    """Compute per-axis mean and std from the training set GT poses."""
    if not hasattr(dataset, "data_list"):
        poses = [dataset[i]["output"][:, 0:3].numpy()
                 for i in tqdm(range(len(dataset)), desc="pose stats")]
        pose_array = np.stack(poses)
    else:
        chunks = []
        last_path, last_gt = None, None
        for item in tqdm(dataset.data_list, desc="pose stats"):
            if item["gt_path"] != last_path:
                last_path = item["gt_path"]
                last_gt   = np.load(last_path)
            if "idx" in item:
                chunks.append(last_gt[item["idx"], :, 0:3])
            else:
                chunks.append(last_gt[:, :, 0:3].reshape(-1, 3))
        first = chunks[0]
        pose_array = (np.stack(chunks) if first.ndim == 2 and first.shape == (17, 3)
                      else np.concatenate(chunks).reshape(-1, 17, 3))

    mean_xyz = pose_array.reshape(-1, 3).mean(axis=0).astype(np.float32)
    std_xyz  = np.maximum(pose_array.reshape(-1, 3).std(axis=0).astype(np.float32), eps)
    return {
        "enabled":               True,
        "mean_xyz":              mean_xyz.tolist(),
        "std_xyz":               std_xyz.tolist(),
        "eps":                   float(eps),
        "num_frames":            int(pose_array.shape[0]),
        "num_joints_per_frame":  int(pose_array.shape[1]),
    }


def make_pose_stats_tensors(pose_stats, device):
    if not pose_stats or not pose_stats.get("enabled", False):
        return None
    return {
        "mean": torch.tensor(pose_stats["mean_xyz"], device=device).view(1, 1, 3),
        "std":  torch.tensor(pose_stats["std_xyz"],  device=device).view(1, 1, 3),
    }


def normalize_pose(pose, stats):
    return pose if stats is None else (pose - stats["mean"]) / stats["std"]


def denormalize_pose(pose, stats):
    return pose if stats is None else pose * stats["std"] + stats["mean"]


# ─── SECTION 4: Training & Evaluation ────────────────────────────────────────

def make_criterion():
    return torch.nn.MSELoss()


def make_optimizer(model, learning_rate):
    return torch.optim.Adam(model.parameters(), lr=learning_rate)


def _batch_to_device(batch, device):
    csi = batch["input_wifi-csi"].to(device).float()
    gt  = batch["output"][:, :, 0:3].to(device).float()
    return csi, gt


def train_one_epoch(model, loader, criterion, optimizer, device, args, epoch, stats=None):
    model.train()
    losses = []
    bar = tqdm(loader, desc=f"train epoch {epoch}", leave=False)
    for i, batch in enumerate(bar):
        if args.max_train_batches is not None and i >= args.max_train_batches:
            break
        csi, gt = _batch_to_device(batch, device)
        pred, _ = model(csi)
        loss = criterion(pred, normalize_pose(gt, stats))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        v = float(loss.item())
        losses.append(v)
        if i % args.log_interval == 0:
            print(f"epoch={epoch} batch={i} loss={v:.6f} "
                  f"lr={optimizer.param_groups[0]['lr']:.6f}", flush=True)
        bar.set_postfix(loss=f"{v:.4f}")
    if not losses:
        raise RuntimeError("No training batches processed.")
    return {"loss": float(np.mean(losses)), "num_batches": len(losses)}


def evaluate(model, loader, criterion, device, stats=None, max_batches=None, desc="eval"):
    model.eval()
    losses, preds, gts = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc=desc, leave=False)):
            if max_batches is not None and i >= max_batches:
                break
            csi, gt = _batch_to_device(batch, device)
            pred, _ = model(csi)
            losses.append(float(criterion(pred, normalize_pose(gt, stats)).item()))
            preds.append(denormalize_pose(pred, stats).detach().cpu().numpy())
            gts.append(gt.detach().cpu().numpy())
    if not losses:
        raise RuntimeError(f"No batches evaluated for '{desc}'.")
    pred_all = np.concatenate(preds)
    gt_all   = np.concatenate(gts)
    metrics  = compute_3d_metrics(pred_all, gt_all)
    metrics.update({"loss": float(np.mean(losses)),
                    "num_samples": int(pred_all.shape[0]),
                    "num_batches": len(losses)})
    return metrics


# ─── SECTION 5: Checkpointing & Logging ──────────────────────────────────────

def _save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _make_training_metadata(args):
    return {
        "profile":                      "phase_C_demo_2_p1s1",
        "architecture_variant":         "dsknet_3d_no_transformer",
        "architecture_source":          "HPE-Li DSKNet / sknet_trans_mmfi.py",
        "source_module":                "model/sknet_trans_mmfi.py",
        "source_output":                "17x2",
        "adapted_output":               "17x3",
        "transformer_enabled":          False,
        "transformer_removed_from_dskconv": True,
        "loss":                         "mse",
        "pose_target_space":            "normalized_xyz",
        "pose_std_eps":                 POSE_STD_EPS,
        "optimizer":                    "adam",
        "learning_rate":                args.lr,
        "weight_decay":                 0.0,
        "lr_scheduler":                 None,
        "grad_clip_norm":               GRAD_CLIP_NORM,
        "training_strategy":            "fixed_epochs",
        "maximum_epochs":                args.epochs,
        "early_stopping":               False,
        "checkpoint_selection_metric":  "val_mpjpe_mm",
        "checkpoint_selection_mode":    "min",
        "test_during_training":          False,
        "final_test_checkpoint":         "best_validation_checkpoint",
    }


def save_checkpoint(path, model, optimizer, epoch, config, args, metrics):
    torch.save({
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "model_name":         MODEL_NAME,
        "model_config":       model.get_model_config(),
        "model_state_dict":   model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch":              epoch,
        "config":             config,
        "args":               vars(args),
        "pose_normalization": metrics.get("pose_normalization"),
        "training_metadata":  _make_training_metadata(args),
        "eval_split_metadata": metrics.get("eval_split_metadata"),
        "metrics":            metrics,
    }, path)


def print_epoch_metrics(epoch, train_m, val_m):
    print(
        f"epoch={epoch} train_loss={train_m['loss']:.6f} "
        f"val_loss={val_m['loss']:.6f} "
        f"val_mpjpe={val_m['mpjpe_mm']:.3f} val_pa_mpjpe={val_m['pa_mpjpe_mm']:.3f} "
        f"pck50mm={val_m['pck_50mm']:.2f}% pck100mm={val_m['pck_100mm']:.2f}% "
        f"g_PCK@50={val_m['g_PCK@50']:.1f}% pa_invalid={val_m['pa_mpjpe_invalid_count']}",
        flush=True,
    )


# ─── SECTION 6: Main ─────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    # ── Config & run directory ────────────────────────────────────────────────
    config = load_config(P1S1_CONFIG_PATH)
    validate_p1s1_config(config)
    config = apply_loader_overrides(config, args)
    run_dir = make_run_dir(args.output_dir, args.run_name, config["split_to_use"])
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    _save_json(run_dir / "args.json", vars(args))

    print(f"run_dir={run_dir} device={device} "
          f"split={config['split_to_use']} dataset={args.dataset_root}", flush=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    (train_loader, val_loader, test_loader,
     train_ds, val_ds, test_ds, split_meta) = make_loaders(
        args.dataset_root, config, args.seed
    )
    print(f"samples: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}", flush=True)
    print(f"batches: train={len(train_loader)} val={len(val_loader)} test={len(test_loader)}", flush=True)

    # Phase C demo 2 trains on train-set XYZ z-scores.
    pose_stats = compute_pose_normalization_stats(train_ds, eps=POSE_STD_EPS)
    print(f"pose_norm: mean={pose_stats['mean_xyz']} std={pose_stats['std_xyz']}", flush=True)
    _save_json(run_dir / "xyz_stats.json", pose_stats)
    stats_t = make_pose_stats_tensors(pose_stats, device)

    # ── Model, loss, optimizer ────────────────────────────────────────────────
    model     = DSKNetMMFI3D().to(device)
    criterion = make_criterion().to(device)
    optimizer = make_optimizer(model, args.lr)
    print(f"model_config={model.get_model_config()} "
          f"loss=mse pose_target=train_xyz_zscore optimizer=adam lr={args.lr} "
          f"training_strategy=fixed_epochs epochs={args.epochs}",
          flush=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_mpjpe     = float("inf")
    best_epoch         = None
    best_state_dict    = None
    history            = []
    best_ckpt  = run_dir / "checkpoints" / "best.pt"
    last_ckpt  = run_dir / "checkpoints" / "last.pt"

    for epoch in range(1, args.epochs + 1):
        train_m = train_one_epoch(model, train_loader, criterion, optimizer,
                                  device, args, epoch, stats=stats_t)
        val_m   = evaluate(model, val_loader, criterion, device,
                           stats=stats_t, max_batches=args.eval_max_batches,
                           desc=f"val epoch {epoch}")

        history.append({"epoch": epoch, "train": train_m, "val": val_m})
        _save_json(run_dir / "history.json", history)
        print_epoch_metrics(epoch, train_m, val_m)

        ckpt_meta = {"train": train_m, "val": val_m,
                     "pose_normalization": pose_stats,
                     "eval_split_metadata": split_meta}
        save_checkpoint(last_ckpt, model, optimizer, epoch, config, args, ckpt_meta)

        cur_mpjpe = val_m["mpjpe_mm"]
        if cur_mpjpe < best_val_mpjpe:
            best_val_mpjpe = cur_mpjpe
            best_epoch     = epoch
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            save_checkpoint(best_ckpt, model, optimizer, epoch, config, args, ckpt_meta)
            print(f"[best] epoch={epoch} val_mpjpe={best_val_mpjpe:.3f}", flush=True)

    if best_state_dict is None:
        raise RuntimeError("Training finished without a best checkpoint.")

    # Final test uses the checkpoint selected by validation MPJPE.
    model.load_state_dict(best_state_dict)
    model.to(device)
    test_m = evaluate(model, test_loader, criterion, device,
                      stats=stats_t, max_batches=args.eval_max_batches,
                      desc="test best-val checkpoint")

    final = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "model_name":         MODEL_NAME,
        "best_epoch":         best_epoch,
        "best_val_mpjpe_mm":  best_val_mpjpe,
        "test_mpjpe_mm":      test_m["mpjpe_mm"],
        "stopped_early":      False,
        "model_config":       model.get_model_config(),
        "pose_normalization": pose_stats,
        "training_metadata":  _make_training_metadata(args),
        "eval_split_metadata": split_meta,
        "history":            history,
        "test":               test_m,
    }

    _save_json(run_dir / "final_metrics.json", final)
    print(
        f"[test] checkpoint=best epoch={best_epoch} "
        f"test_mpjpe={test_m['mpjpe_mm']:.3f} "
        f"test_pa_mpjpe={test_m['pa_mpjpe_mm']:.3f}",
        flush=True,
    )
    print(f"[done] run_dir={run_dir}", flush=True)


if __name__ == "__main__":
    main()
