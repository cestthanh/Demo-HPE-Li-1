"""
tools/evaluate.py
Offline evaluation of a saved DSKNetMMFI3D checkpoint.

Usage
-----
    python tools/evaluate.py \\
        --checkpoint checkpoints/phase_C_seed_0.pt \\
        --dataset-root /path/to/mmfi/dataset \\
        --eval-split test

Output files (written next to the checkpoint by default):
    metrics_test.json                 - full metrics dict
    per_joint_mpjpe_test.md           - per-joint MPJPE table
    graphpose_benchmark_test.md       - g_PCK / MPJPE benchmark row
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feeder import make_dataloader, make_dataset
from feeder.splits import split_eval_dataset_by_sequence
from model.dsknet3d import (
    MODEL_NAME,
    DSKNetMMFI3D,
    get_model_config_from_checkpoint,
    normalize_state_dict,
)
from utils.eval_3d import MMFI_17_JOINT_NAMES, compute_3d_metrics


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a DSKNetMMFI3D checkpoint on MMFi."
    )
    p.add_argument("--checkpoint",  required=True,
                   help="Path to .pt checkpoint file.")
    p.add_argument("--dataset-root",
                   default=os.getenv("MMFI_DATASET_ROOT",
                                     str(PROJECT_ROOT / "data" / "mmfi" / "dataset")),
                   help="Root directory of the MMFi dataset.")
    p.add_argument("--config",
                   default=str(PROJECT_ROOT / "config" / "mmfi" /
                               "config_phase_c_demo_2_p1s1.yaml"),
                   help="Fallback YAML config if checkpoint has none.")
    p.add_argument("--split-to-use", default=None,
                   choices=["random_split", "cross_scene_split",
                             "cross_subject_split", "manual_split"])
    p.add_argument("--eval-split",   default="test",
                   choices=["val", "test", "eval_all"])
    p.add_argument("--eval-partition-unit", default="auto",
                   choices=["auto", "sequence", "frame"],
                   help="'auto' reads checkpoint metadata (defaults to 'frame' for old ckpts).")
    p.add_argument("--batch-size",   type=int,  default=8)
    p.add_argument("--num-workers",  type=int,  default=4)
    p.add_argument("--device",       default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed",         type=int,  default=0)
    p.add_argument("--max-batches",  type=int,  default=None)
    p.add_argument("--output-json",          default=None)
    p.add_argument("--output-md",            default=None)
    p.add_argument("--output-graphpose-md",  default=None)
    p.add_argument("--method-name",  default=MODEL_NAME)
    return p.parse_args()


# ─── Setup helpers ────────────────────────────────────────────────────────────

def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_arg)


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_config(args, checkpoint):
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        config = copy.deepcopy(checkpoint["config"])
    else:
        with open(args.config, encoding="utf-8") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    if args.split_to_use is not None:
        config["split_to_use"] = args.split_to_use
    for key in ("train_loader", "val_loader", "test_loader"):
        config[key] = dict(config[key])
        config[key]["num_workers"] = args.num_workers
    config["val_loader"]["batch_size"]  = args.batch_size
    config["test_loader"]["batch_size"] = args.batch_size
    return config


def get_partition_unit(checkpoint, requested):
    if requested != "auto":
        return requested
    if isinstance(checkpoint, dict):
        meta = checkpoint.get("eval_split_metadata") or \
               (checkpoint.get("metrics") or {}).get("eval_split_metadata")
        if isinstance(meta, dict) and meta.get("split_unit") == "sequence":
            return "sequence"
    return "frame"


# ─── Data ─────────────────────────────────────────────────────────────────────

def make_eval_loader(dataset_root, config, args, partition_unit):
    _, eval_dataset = make_dataset(dataset_root, config)

    if args.eval_split == "eval_all":
        selected = eval_dataset
        meta = {"split_unit": "none", "selection": "eval_all",
                "num_frames": len(eval_dataset)}
    elif partition_unit == "sequence":
        val_ds, test_ds, meta = split_eval_dataset_by_sequence(
            eval_dataset, test_size=0.5, random_state=41
        )
        selected = val_ds if args.eval_split == "val" else test_ds
        meta = {**meta, "selection": args.eval_split}
    else:
        val_idx, test_idx = train_test_split(
            list(range(len(eval_dataset))), test_size=0.5, random_state=41
        )
        chosen = val_idx if args.eval_split == "val" else test_idx
        selected = Subset(eval_dataset, chosen)
        meta = {"split_unit": "frame", "selection": args.eval_split,
                "test_size": 0.5, "random_state": 41, "num_frames": len(chosen)}

    loader = make_dataloader(
        selected, is_training=False,
        generator=torch.Generator().manual_seed(args.seed),
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=config["test_loader"].get("pin_memory", False),
    )
    return loader, selected, meta


# ─── Model loading ────────────────────────────────────────────────────────────

def load_model(checkpoint, device):
    if isinstance(checkpoint, dict):
        checkpoint_model = checkpoint.get("model_name")
        if checkpoint_model not in (None, MODEL_NAME):
            raise ValueError(
                f"Checkpoint model_name={checkpoint_model!r} is incompatible "
                f"with {MODEL_NAME!r}."
            )
        sd = checkpoint.get("model_state_dict")
        if not isinstance(sd, dict):
            raise ValueError("Checkpoint missing 'model_state_dict'.")
        sd = normalize_state_dict(sd)
    else:
        raise ValueError("Unsupported checkpoint format.")
    cfg = get_model_config_from_checkpoint(checkpoint)
    model = DSKNetMMFI3D(**cfg).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model


# ─── Pose normalisation ───────────────────────────────────────────────────────

def get_pose_normalization(checkpoint):
    if isinstance(checkpoint, dict):
        stats = checkpoint.get("pose_normalization") or \
                (checkpoint.get("metrics") or {}).get("pose_normalization")
        if stats:
            return stats
    return {"enabled": False}


def make_pose_stats_tensors(pose_stats, device):
    if not pose_stats or not pose_stats.get("enabled", False):
        return None
    return {
        "mean": torch.tensor(pose_stats["mean_xyz"], device=device).view(1, 1, 3),
        "std":  torch.tensor(pose_stats["std_xyz"],  device=device).view(1, 1, 3),
    }


def denormalize_pose(pose, stats):
    if stats is None:
        return pose
    return pose * stats["std"] + stats["mean"]


# ─── Evaluation loop ──────────────────────────────────────────────────────────

def evaluate(model, loader, device, pose_stats=None, max_batches=None):
    preds, gts = [], []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="evaluate")):
            if max_batches is not None and i >= max_batches:
                break
            csi  = batch["input_wifi-csi"].to(device).float()
            gt   = batch["output"][:, :, 0:3].to(device).float()
            pred, _ = model(csi)
            pred = denormalize_pose(pred, pose_stats)
            preds.append(pred.detach().cpu().numpy())
            gts.append(gt.detach().cpu().numpy())

    if not preds:
        raise RuntimeError("No batches evaluated.")

    pred_all = np.concatenate(preds)
    gt_all   = np.concatenate(gts)
    metrics  = compute_3d_metrics(pred_all, gt_all)
    metrics["num_samples"] = int(pred_all.shape[0])
    metrics["num_batches"] = int(len(preds))
    return metrics


# ─── Report rendering ─────────────────────────────────────────────────────────

def _per_joint_md(metrics):
    rows = ["| Joint | MPJPE (mm) |", "|---|---:|"]
    for name, val in zip(MMFI_17_JOINT_NAMES, metrics["per_joint_mpjpe_mm"]):
        rows.append(f"| {name} | {val:.3f} |")
    rows.append(f"| **Average** | **{metrics['mpjpe_mm']:.3f}** |")
    return "\n".join(rows) + "\n"


def _graphpose_md(metrics, method_name):
    lines = [
        "| Method | g_PCK@10 | g_PCK@20 | g_PCK@30 | g_PCK@40 | g_PCK@50 | MPJPE | PA-MPJPE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (f"| {method_name} "
         f"| {metrics['g_PCK@10']:.1f} "
         f"| {metrics['g_PCK@20']:.1f} "
         f"| {metrics['g_PCK@30']:.1f} "
         f"| {metrics['g_PCK@40']:.1f} "
         f"| {metrics['g_PCK@50']:.1f} "
         f"| {metrics['mpjpe_mm']:.1f} "
         f"| {metrics['pa_mpjpe_mm']:.1f} |"),
        "",
        "`g_PCK@N` thresholds are N/100 × GT body scale (R.Hip→L.Shoulder), not millimetres.",
    ]
    return "\n".join(lines) + "\n"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args        = parse_args()
    device      = resolve_device(args.device)
    ckpt_path   = Path(args.checkpoint)
    checkpoint  = load_checkpoint(ckpt_path, device)
    config      = load_config(args, checkpoint)
    unit        = get_partition_unit(checkpoint, args.eval_partition_unit)
    loader, selected_ds, split_meta = make_eval_loader(
        args.dataset_root, config, args, unit
    )
    model       = load_model(checkpoint, device)
    pose_stats  = get_pose_normalization(checkpoint)
    stats_t     = make_pose_stats_tensors(pose_stats, device)

    metrics = evaluate(model, loader, device, pose_stats=stats_t,
                       max_batches=args.max_batches)
    metrics.update({
        "checkpoint":         str(ckpt_path),
        "dataset_root":       args.dataset_root,
        "split_to_use":       config["split_to_use"],
        "eval_split":         args.eval_split,
        "model_name":         MODEL_NAME,
        "model_config":       model.get_model_config(),
        "pose_normalization": pose_stats,
        "eval_split_metadata": split_meta,
    })

    # ── Output paths ─────────────────────────────────────────────────────────
    out_json = Path(args.output_json) if args.output_json \
               else ckpt_path.parent / f"metrics_{args.eval_split}.json"
    out_md   = Path(args.output_md)   if args.output_md \
               else ckpt_path.parent / f"per_joint_mpjpe_{args.eval_split}.md"
    out_gmd  = Path(args.output_graphpose_md) if args.output_graphpose_md \
               else ckpt_path.parent / f"graphpose_benchmark_{args.eval_split}.md"

    for p in (out_json, out_md, out_gmd):
        p.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    out_md.write_text(_per_joint_md(metrics), encoding="utf-8")
    out_gmd.write_text(
        _graphpose_md(metrics, args.method_name),
        encoding="utf-8",
    )

    # ── Console summary ───────────────────────────────────────────────────────
    m = metrics
    print(
        f"[eval_split={args.eval_split}] samples={len(selected_ds)} "
        f"mpjpe={m['mpjpe_mm']:.3f} pa_mpjpe={m['pa_mpjpe_mm']:.3f} "
        f"pck50mm={m['pck_50mm']:.2f}% pck100mm={m['pck_100mm']:.2f}% "
        f"g_PCK@50={m['g_PCK@50']:.1f}% "
        f"normalize_pose={pose_stats.get('enabled', False)} "
        f"model_config={model.get_model_config()}",
        flush=True,
    )
    print(f"[saved] {out_json}", flush=True)
    print(f"[saved] {out_md}",   flush=True)
    print(f"[saved] {out_gmd}",  flush=True)


if __name__ == "__main__":
    main()
