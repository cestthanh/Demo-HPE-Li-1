"""
train.py
Training entry point for DSKNetMMFI3D - the DSK-only 3D HPE ablation
from WiFi-CSI signals on the MMFi dataset.

Typical usage
-------------
    # Default (random split, 20 epochs)
    python train.py

    # Cross-subject split, 60 epochs on GPU
    python train.py --split-to-use cross_subject_split --epochs 60 --device cuda

    # Quick smoke test (2 epochs, 10 batches each)
    python train.py --epochs 2 --max-train-batches 10 --eval-max-batches 5

Environment variables
---------------------
    MMFI_DATASET_ROOT            Path to MMFi dataset root.
    DSK_ONLY_OUTPUT              Output directory for runs.
    DSK_ONLY_EPOCHS              Number of epochs.
    DSK_ONLY_LR                  Learning rate.
    DSK_ONLY_WEIGHT_DECAY        Optimizer weight decay.
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
sys.path.insert(0, str(PROJECT_ROOT))

from feeder import make_dataloader, make_dataset
from feeder.splits import split_eval_dataset_by_sequence
from model.dsknet3d import CHECKPOINT_FORMAT_VERSION, DSKNetMMFI3D
from utils.eval_3d import compute_3d_metrics


# ─── SECTION 1: CLI & Setup ───────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train the DSK-only 3D HPE ablation on MMFi."
    )
    # Data
    p.add_argument("--config",
                   default=str(PROJECT_ROOT / "config" / "mmfi" /
                               "config_p1s1.yaml"),
                   help="Path to the MMFi YAML config.")
    p.add_argument("--dataset-root",
                   default=os.getenv("MMFI_DATASET_ROOT",
                                     str(PROJECT_ROOT / "data" / "mmfi" / "dataset")),
                   help="MMFi dataset root directory.")
    p.add_argument("--split-to-use", default=None,
                   choices=["random_split", "cross_scene_split",
                             "cross_subject_split", "manual_split"],
                   help="Override config split_to_use without editing YAML.")
    # Output
    p.add_argument("--output-dir",
                   default=os.getenv("DSK_ONLY_OUTPUT",
                                     str(PROJECT_ROOT / "results" / "dsk_only")),
                   help="Root directory for logs, metrics, and checkpoints.")
    p.add_argument("--run-name", default=None,
                   help="Optional run name (default: timestamp + split).")
    # Training
    p.add_argument("--epochs",       type=int,   default=int(os.getenv("DSK_ONLY_EPOCHS", "20")))
    p.add_argument("--lr",           type=float, default=float(os.getenv("DSK_ONLY_LR", "0.001")))
    p.add_argument("--optimizer",    default="adam", choices=["adam", "adamw"])
    p.add_argument("--weight-decay", type=float, default=float(os.getenv("DSK_ONLY_WEIGHT_DECAY", "0.0")))
    p.add_argument("--seed",         type=int,   default=0)
    p.add_argument("--device",       default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--loss",         default="smooth_l1", choices=["smooth_l1", "mse"])
    p.add_argument("--smooth-l1-beta", type=float, default=0.05)
    p.add_argument("--grad-clip",      type=float, default=1.0)
    p.add_argument("--log-interval",   type=int,   default=100)
    # Pose normalisation
    p.add_argument("--normalize-pose", action="store_true",
                   help="Z-score normalise pose targets during training.")
    p.add_argument("--pose-std-eps", type=float, default=1e-6)
    # Early stopping
    p.add_argument("--early-stopping-patience",  type=int,   default=None)
    p.add_argument("--early-stopping-min-delta", type=float, default=1.0)
    # Debug / batch limits
    p.add_argument("--eval-max-batches",  type=int, default=None)
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--train-batch-size",  type=int, default=None)
    p.add_argument("--val-batch-size",    type=int, default=None)
    p.add_argument("--test-batch-size",   type=int, default=None)
    p.add_argument("--num-workers",       type=int, default=None)
    p.add_argument("--no-test", action="store_true",
                   help="Skip final test evaluation on best checkpoint.")
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
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def apply_loader_overrides(config, args):
    config = copy.deepcopy(config)
    if args.split_to_use is not None:
        config["split_to_use"] = args.split_to_use
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

def make_criterion(args):
    return (torch.nn.SmoothL1Loss(beta=args.smooth_l1_beta)
            if args.loss == "smooth_l1" else torch.nn.MSELoss())


def make_optimizer(args, model):
    cls = torch.optim.Adam if args.optimizer == "adam" else torch.optim.AdamW
    return cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


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
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
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
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _make_training_metadata(args):
    return {
        "architecture_variant":         "hpe_li_dsk_only_3d_ablation",
        "transformer_enabled":          False,
        "ablation_reference":           "HPE-Li-3D DSKNetTransMMFI3D",
        "loss":                         args.loss,
        "pose_target_space":            "normalized_xyz" if args.normalize_pose else "metric_xyz_meters",
        "checkpoint_selection_metric":  "val_mpjpe_mm",
        "checkpoint_selection_mode":    "min",
    }


def save_checkpoint(path, model, optimizer, epoch, config, args, metrics):
    torch.save({
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "model_name":         "DSKNetMMFI3D",
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
    print(
        f"  [diag] axis_mae={val_m['axis_mae_mm_by_name']} "
        f"root_mpjpe={val_m['root_mpjpe_mm']:.3f} "
        f"rc_mpjpe={val_m['root_centered_mpjpe_mm']:.3f} "
        f"const_mpjpe={val_m['constant_mean_pose_mpjpe_mm']:.3f} "
        f"pa_gain={val_m['pa_mpjpe_gain_over_constant_mean_pose_mm']:.3f}",
        flush=True,
    )


# ─── SECTION 6: Main ─────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    # ── Config & run directory ────────────────────────────────────────────────
    config  = apply_loader_overrides(load_config(args.config), args)
    run_dir = make_run_dir(args.output_dir, args.run_name, config["split_to_use"])
    with open(run_dir / "config.yaml", "w") as f:
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

    # ── Pose normalisation ────────────────────────────────────────────────────
    pose_stats = {"enabled": False}
    if args.normalize_pose:
        pose_stats = compute_pose_normalization_stats(train_ds, eps=args.pose_std_eps)
        print(f"pose_norm: mean={pose_stats['mean_xyz']} std={pose_stats['std_xyz']}", flush=True)
    _save_json(run_dir / "xyz_stats.json", pose_stats)
    stats_t = make_pose_stats_tensors(pose_stats, device)

    # ── Model, loss, optimizer ────────────────────────────────────────────────
    model     = DSKNetMMFI3D().to(device)
    criterion = make_criterion(args).to(device)
    optimizer = make_optimizer(args, model)
    print(f"model_config={model.get_model_config()} "
          f"optimizer={args.optimizer} lr={args.lr}", flush=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_mpjpe     = float("inf")
    best_epoch         = None
    no_improve_count   = 0
    stopped_early      = False
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
            prev_best      = best_val_mpjpe
            best_val_mpjpe = cur_mpjpe
            best_epoch     = epoch
            save_checkpoint(best_ckpt, model, optimizer, epoch, config, args, ckpt_meta)
            print(f"[best] epoch={epoch} val_mpjpe={best_val_mpjpe:.3f}", flush=True)
            if cur_mpjpe < prev_best - args.early_stopping_min_delta or prev_best == float("inf"):
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1

        if (args.early_stopping_patience and
                no_improve_count >= args.early_stopping_patience):
            print(f"[early stop] epoch={epoch} best_epoch={best_epoch} "
                  f"best_val_mpjpe={best_val_mpjpe:.3f}", flush=True)
            stopped_early = True
            break

    # ── Final test evaluation on best checkpoint ──────────────────────────────
    final = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "model_name":         "DSKNetMMFI3D",
        "best_epoch":         best_epoch,
        "best_val_mpjpe_mm":  best_val_mpjpe,
        "stopped_early":      stopped_early,
        "model_config":       model.get_model_config(),
        "pose_normalization": pose_stats,
        "training_metadata":  _make_training_metadata(args),
        "eval_split_metadata": split_meta,
        "history":            history,
    }

    if not args.no_test:
        try:
            ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        test_m = evaluate(model, test_loader, criterion, device,
                          stats=stats_t, max_batches=args.eval_max_batches,
                          desc="test best")
        final["test"] = test_m
        print(
            f"[test] loss={test_m['loss']:.6f} "
            f"mpjpe={test_m['mpjpe_mm']:.3f} pa_mpjpe={test_m['pa_mpjpe_mm']:.3f} "
            f"pck50mm={test_m['pck_50mm']:.2f}% "
            f"g_PCK@50={test_m['g_PCK@50']:.1f}%",
            flush=True,
        )

    _save_json(run_dir / "final_metrics.json", final)
    print(f"[done] run_dir={run_dir}", flush=True)


if __name__ == "__main__":
    main()
