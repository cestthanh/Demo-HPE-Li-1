"""
utils/eval_3d.py
3D pose evaluation metrics for the MMFi 17-joint benchmark.

Metrics
-------
- MPJPE (mm)         : Mean Per-Joint Position Error in millimetres.
- PA-MPJPE (mm)      : Procrustes-Aligned MPJPE (scale + rotation corrected).
- PCK@50mm / @100mm  : % joints within fixed distance threshold.
- g_PCK@{10..50}     : Body-scale PCK at thresholds 0.1–0.5 × GT body scale.
                        Body scale = GT distance R.Hip (1) → L.Shoulder (11).

Note: g_PCK thresholds are body-scale ratios, NOT millimetres.
These values are not comparable to legacy GraphPose-Fi results that
used joint indices (5, 12) for the body scale.
"""
import math

import numpy as np


# ─── Constants ────────────────────────────────────────────────────────────────

MMFI_17_JOINT_NAMES = [
    "Bot Torso",
    "R.Hip",      "R.Knee",   "R.Foot",
    "L.Hip",      "L.Knee",   "L.Foot",
    "Center Torso", "Upper Torso", "Neck Base", "Center Head",
    "L.Shoulder", "L.Elbow",  "L.Hand",
    "R.Shoulder", "R.Elbow",  "R.Hand",
]

# Body-scale joint pair used for g_PCK (corrected MMFi definition)
MMFI_BODY_SCALE_JOINTS      = (1, 11)          # R.Hip → L.Shoulder
MMFI_BODY_SCALE_JOINT_NAMES = ("R.Hip", "L.Shoulder")
GRAPHPOSE_PCK_THRESHOLDS    = (0.1, 0.2, 0.3, 0.4, 0.5)

# Backward-compatible alias
GRAPHPOSE_MMFI_SCALE_JOINTS = MMFI_BODY_SCALE_JOINTS


# ─── Array helpers ────────────────────────────────────────────────────────────

def _to_numpy(array):
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    return np.asarray(array, dtype=np.float64)


def _validate(pred, gt):
    pred, gt = _to_numpy(pred), _to_numpy(gt)
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, gt={gt.shape}")
    if pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError(
            f"Expected (batch, joints, 3), got {pred.shape}"
        )
    return pred, gt


# ─── Individual metrics ───────────────────────────────────────────────────────

def mpjpe_mm(pred_xyz, gt_xyz):
    """Mean Per-Joint Position Error in mm."""
    pred, gt = _validate(pred_xyz, gt_xyz)
    return float(np.mean(np.linalg.norm(pred - gt, axis=-1)) * 1000.0)


def per_joint_mpjpe_mm(pred_xyz, gt_xyz):
    """Per-joint MPJPE in mm, returns a list of length=num_joints."""
    pred, gt = _validate(pred_xyz, gt_xyz)
    return (np.mean(np.linalg.norm(pred - gt, axis=-1), axis=0) * 1000.0).tolist()


def pck_3d_mm(pred_xyz, gt_xyz, threshold_mm=50.0):
    """Percentage of joints within a fixed distance threshold (mm)."""
    pred, gt = _validate(pred_xyz, gt_xyz)
    dist_mm = np.linalg.norm(pred - gt, axis=-1) * 1000.0
    return float(np.mean(dist_mm <= threshold_mm) * 100.0)


def body_scale_mmfi(gt_xyz, scale_joints=MMFI_BODY_SCALE_JOINTS):
    """GT body scale: Euclidean distance between R.Hip and L.Shoulder per frame."""
    gt = _to_numpy(gt_xyz)
    a, b = scale_joints
    return np.linalg.norm(gt[:, a, :] - gt[:, b, :], axis=-1)


def graphpose_pck_mmfi(pred_xyz, gt_xyz, threshold, eps=1e-8):
    """Body-scale normalised PCK (g_PCK).

    ``threshold=0.5`` means joint error ≤ 0.5 × GT body scale.
    """
    pred, gt = _validate(pred_xyz, gt_xyz)
    if pred.shape[1] <= max(MMFI_BODY_SCALE_JOINTS):
        raise ValueError(
            f"g_PCK needs ≥{max(MMFI_BODY_SCALE_JOINTS)+1} joints, got {pred.shape[1]}"
        )
    dist  = np.linalg.norm(pred - gt, axis=-1)
    scale = body_scale_mmfi(gt)
    dist_norm = np.divide(dist, scale[:, None],
                          out=np.full_like(dist, np.inf),
                          where=scale[:, None] > eps)
    return float(np.mean(dist_norm <= threshold) * 100.0)


def _procrustes_align(pred, gt, eps=1e-8):
    """Align pred to gt via optimal similarity transform (Procrustes).

    Returns aligned prediction or None if the alignment is degenerate.
    """
    if not (np.isfinite(pred).all() and np.isfinite(gt).all()):
        return None
    mu_gt, mu_pred = gt.mean(0), pred.mean(0)
    gt_c, pred_c   = gt - mu_gt, pred - mu_pred
    norm_gt, norm_pred = np.sqrt((gt_c**2).sum()), np.sqrt((pred_c**2).sum())
    if norm_gt < eps or norm_pred < eps:
        return None
    try:
        u, s, vt = np.linalg.svd((gt_c / norm_gt).T @ (pred_c / norm_pred),
                                   full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    v = vt.T
    R = v @ u.T
    if np.linalg.det(R) < 0:
        v[:, -1] *= -1
        s[-1]    *= -1
        R = v @ u.T
    scale = s.sum() * norm_gt / norm_pred
    t = mu_gt - scale * (mu_pred @ R)
    return scale * (pred @ R) + t


def pa_mpjpe_mm(pred_xyz, gt_xyz, return_invalid_count=False):
    """Procrustes-Aligned MPJPE in mm."""
    pred, gt = _validate(pred_xyz, gt_xyz)
    errors, invalid = [], 0
    for p, g in zip(pred, gt):
        aligned = _procrustes_align(p, g)
        if aligned is None:
            invalid += 1
            continue
        errors.append(np.linalg.norm(aligned - g, axis=-1).mean() * 1000.0)
    value = float(np.mean(errors)) if errors else math.nan
    return (value, invalid) if return_invalid_count else value


# ─── Aggregate metric bundle ──────────────────────────────────────────────────

def compute_3d_metrics(pred_xyz, gt_xyz):
    """Compute the metrics used by the P1-S1 training/evaluation pipeline."""
    pred, gt = _validate(pred_xyz, gt_xyz)
    per_joint = per_joint_mpjpe_mm(pred, gt)
    pa_value, pa_invalid = pa_mpjpe_mm(
        pred, gt, return_invalid_count=True
    )

    metrics = {
        "mpjpe_mm":                    mpjpe_mm(pred, gt),
        "pa_mpjpe_mm":                 pa_value,
        "pa_mpjpe_invalid_count":      pa_invalid,
        "pck_50mm":                    pck_3d_mm(pred, gt, 50.0),
        "pck_100mm":                   pck_3d_mm(pred, gt, 100.0),
        "per_joint_mpjpe_mm":          per_joint,
        "per_joint_mpjpe_mm_by_name":  dict(
            zip(MMFI_17_JOINT_NAMES, per_joint)
        ),
        "g_PCK_scale_joints":          list(MMFI_BODY_SCALE_JOINTS),
        "g_PCK_scale_joint_names":     list(MMFI_BODY_SCALE_JOINT_NAMES),
        "g_PCK_scale_source":          "ground_truth",
    }

    for thr in GRAPHPOSE_PCK_THRESHOLDS:
        tag = int(round(thr * 100))
        metrics[f"g_PCK@{tag}"] = graphpose_pck_mmfi(pred, gt, thr)

    return metrics
