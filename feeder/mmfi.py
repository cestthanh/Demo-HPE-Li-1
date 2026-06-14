"""
feeder/mmfi.py
MMFi dataset loader — parses WiFi-CSI + ground-truth 3D pose from disk.

Source: ECCV 2024 HPE-Li (phase_C_demo_2), dataset_lib/mmfi.py
"""
import glob
import os

import cv2
import numpy as np
import scipy.io as scio
import torch
from torch.utils.data import DataLoader, Dataset


# ─── Config Decoding ──────────────────────────────────────────────────────────

def decode_config(config):
    """Convert a YAML config dict into per-subject/action train & val dicts."""
    all_subjects = [
        'S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10',
        'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20',
        'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30',
        'S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40',
    ]
    all_actions = [
        'A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10',
        'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20',
        'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27',
    ]

    # Protocol: limit action set
    if config['protocol'] == 'protocol1':    # Daily actions
        actions = ['A02', 'A03', 'A04', 'A05', 'A13', 'A14', 'A17', 'A18',
                   'A19', 'A20', 'A21', 'A22', 'A23', 'A27']
    elif config['protocol'] == 'protocol2':  # Rehabilitation actions
        actions = ['A01', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12',
                   'A15', 'A16', 'A24', 'A25', 'A26']
    else:                                    # protocol3 — all actions
        actions = all_actions

    train_form = {}
    val_form = {}

    split = config['split_to_use']

    if split == 'random_split':
        rs = config['random_split']['random_seed']
        ratio = config['random_split']['ratio']
        for action in actions:
            np.random.seed(rs)
            idx = np.random.permutation(len(all_subjects))
            idx_train = idx[:int(np.floor(ratio * len(all_subjects)))]
            idx_val = idx[int(np.floor(ratio * len(all_subjects))):]
            subjects_train = np.array(all_subjects)[idx_train].tolist()
            subjects_val = np.array(all_subjects)[idx_val].tolist()
            for subject in all_subjects:
                if subject in subjects_train:
                    train_form.setdefault(subject, []).append(action)
                if subject in subjects_val:
                    val_form.setdefault(subject, []).append(action)
            rs += 1

    elif split == 'cross_scene_split':
        subjects_train = [f'S{i:02d}' for i in range(1, 31)]
        subjects_val = [f'S{i:02d}' for i in range(31, 41)]
        for s in subjects_train:
            train_form[s] = actions
        for s in subjects_val:
            val_form[s] = actions

    elif split == 'cross_subject_split':
        subjects_train = config['cross_subject_split']['train_dataset']['subjects']
        subjects_val = config['cross_subject_split']['val_dataset']['subjects']
        for s in subjects_train:
            train_form[s] = actions
        for s in subjects_val:
            val_form[s] = actions

    else:  # manual_split
        subjects_train = config['manual_split']['train_dataset']['subjects']
        subjects_val = config['manual_split']['val_dataset']['subjects']
        actions_train = config['manual_split']['train_dataset']['actions']
        actions_val = config['manual_split']['val_dataset']['actions']
        for s in subjects_train:
            train_form[s] = actions_train
        for s in subjects_val:
            val_form[s] = actions_val

    return {
        'train_dataset': {'modality': config['modality'], 'split': 'training',   'data_form': train_form},
        'val_dataset':   {'modality': config['modality'], 'split': 'validation', 'data_form': val_form},
    }


# ─── Database Index ───────────────────────────────────────────────────────────

class MMFi_Database:
    """Builds an in-memory index of all scene/subject/action/modality paths."""

    def __init__(self, data_root):
        self.data_root = data_root
        self.scenes = {}
        self.subjects = {}
        self.actions = {}
        self.modalities = {}
        self._build_index()

    def _build_index(self):
        for scene in sorted(os.listdir(self.data_root)):
            scene_path = os.path.join(self.data_root, scene)
            if scene.startswith('.') or not os.path.isdir(scene_path):
                continue
            self.scenes[scene] = {}
            for subject in sorted(os.listdir(scene_path)):
                subject_path = os.path.join(scene_path, subject)
                if subject.startswith('.') or not os.path.isdir(subject_path):
                    continue
                self.scenes[scene][subject] = {}
                self.subjects[subject] = {}
                for action in sorted(os.listdir(subject_path)):
                    action_path = os.path.join(subject_path, action)
                    if action.startswith('.') or not os.path.isdir(action_path):
                        continue
                    self.scenes[scene][subject][action] = {}
                    self.subjects[subject][action] = {}
                    self.actions.setdefault(action, {}).setdefault(scene, {})[subject] = {}
                    for modality in ['infra1', 'infra2', 'depth', 'rgb',
                                     'lidar', 'mmwave', 'wifi-csi']:
                        data_path = os.path.join(
                            self.data_root, scene, subject, action, modality
                        )
                        self.scenes[scene][subject][action][modality] = data_path
                        self.subjects[subject][action][modality] = data_path
                        self.actions[action][scene][subject][modality] = data_path
                        self.modalities.setdefault(modality, {}) \
                                       .setdefault(scene, {}) \
                                       .setdefault(subject, {})[action] = data_path


# ─── Dataset ──────────────────────────────────────────────────────────────────

_SUBJECT_TO_SCENE = {
    **{f'S{i:02d}': 'E01' for i in range(1, 11)},
    **{f'S{i:02d}': 'E02' for i in range(11, 21)},
    **{f'S{i:02d}': 'E03' for i in range(21, 31)},
    **{f'S{i:02d}': 'E04' for i in range(31, 41)},
}

_MODALITY_EXT = {
    'rgb': '.npy', 'infra1': '.npy', 'infra2': '.npy',
    'lidar': '.bin', 'mmwave': '.bin',
    'depth': '.png',
    'wifi-csi': '.mat',
}

_FRAMES_PER_SEQUENCE = 297


class MMFi_Dataset(Dataset):
    """PyTorch Dataset for the MMFi multi-modal WiFi pose benchmark."""

    def __init__(self, data_base, data_unit, modality, split, data_form):
        self.data_base = data_base
        self.data_unit = data_unit
        self.modality = modality.split('|') if isinstance(modality, str) else list(modality)
        for m in self.modality:
            assert m in _MODALITY_EXT, f"Unsupported modality: {m}"
        self.split = split
        self.data_source = data_form
        self.data_list = self._build_data_list()

    # ── Internal helpers ──

    @staticmethod
    def _get_scene(subject):
        scene = _SUBJECT_TO_SCENE.get(subject)
        if scene is None:
            raise ValueError(f"Subject {subject!r} not found in MMFi.")
        return scene

    def _gt_path(self, subject, action):
        scene = self._get_scene(subject)
        return os.path.join(
            self.data_base.data_root, scene, subject, action, 'ground_truth.npy'
        )

    def _frame_path(self, subject, action, mod, frame_idx):
        scene = self._get_scene(subject)
        ext = _MODALITY_EXT[mod]
        return os.path.join(
            self.data_base.data_root, scene, subject, action,
            mod, f"frame{frame_idx + 1:03d}{ext}"
        )

    def _build_data_list(self):
        data_info = []
        for subject, actions in self.data_source.items():
            for action in actions:
                gt = self._gt_path(subject, action)
                if self.data_unit == 'sequence':
                    entry = {
                        'modality': self.modality,
                        'scene': self._get_scene(subject),
                        'subject': subject, 'action': action,
                        'gt_path': gt,
                    }
                    for mod in self.modality:
                        entry[mod + '_path'] = os.path.join(
                            self.data_base.data_root,
                            self._get_scene(subject), subject, action, mod
                        )
                    data_info.append(entry)
                elif self.data_unit == 'frame':
                    for idx in range(_FRAMES_PER_SEQUENCE):
                        entry = {
                            'modality': self.modality,
                            'scene': self._get_scene(subject),
                            'subject': subject, 'action': action,
                            'gt_path': gt, 'idx': idx,
                        }
                        valid = True
                        for mod in self.modality:
                            fp = self._frame_path(subject, action, mod, idx)
                            entry[mod + '_path'] = fp
                            if os.path.getsize(fp) == 0:
                                valid = False
                        if valid:
                            data_info.append(entry)
                else:
                    raise ValueError(f"Unsupported data_unit: {self.data_unit!r}")
        return data_info

    # ── I/O helpers ──

    def _read_dir(self, dir_path):
        _, mod = os.path.split(dir_path)
        data = []
        if mod in ('infra1', 'infra2', 'rgb'):
            for f in sorted(glob.glob(os.path.join(dir_path, 'frame*.npy'))):
                data.append(np.load(f))
            return np.array(data)
        if mod == 'depth':
            for img in sorted(glob.glob(os.path.join(dir_path, 'frame*.png'))):
                frame = cv2.imread(img, cv2.IMREAD_UNCHANGED) * 0.001
                data.append(frame)
            return np.array(data)
        if mod == 'lidar':
            for bf in sorted(glob.glob(os.path.join(dir_path, 'frame*.bin'))):
                with open(bf, 'rb') as f:
                    raw = f.read()
                data.append(np.frombuffer(raw, dtype=np.float64).reshape(-1, 3))
            return data
        if mod == 'mmwave':
            for bf in sorted(glob.glob(os.path.join(dir_path, 'frame*.bin'))):
                with open(bf, 'rb') as f:
                    raw = f.read()
                data.append(np.frombuffer(raw, dtype=np.float64).copy().reshape(-1, 5))
            return data
        if mod == 'wifi-csi':
            for mat_file in sorted(glob.glob(os.path.join(dir_path, 'frame*.mat'))):
                data.append(self._load_csi_mat(mat_file))
            return np.array(data)
        raise ValueError(f"Unseen modality: {mod!r}")

    def _read_frame(self, frame_path):
        _mod_dir, _ = os.path.split(frame_path)
        _, mod = os.path.split(_mod_dir)
        if mod in ('infra1', 'infra2', 'rgb'):
            return np.load(frame_path)
        if mod == 'depth':
            return cv2.imread(frame_path, cv2.IMREAD_UNCHANGED) * 0.001
        if mod == 'lidar':
            with open(frame_path, 'rb') as f:
                raw = f.read()
            return np.frombuffer(raw, dtype=np.float64).reshape(-1, 3)
        if mod == 'mmwave':
            with open(frame_path, 'rb') as f:
                raw = f.read()
            return np.frombuffer(raw, dtype=np.float64).copy().reshape(-1, 5)
        if mod == 'wifi-csi':
            return self._load_csi_mat(frame_path)
        raise ValueError(f"Unseen modality: {mod!r}")

    @staticmethod
    def _load_csi_mat(path):
        """Load a WiFi-CSI .mat file, impute NaN/Inf, and min-max normalise."""
        mat = scio.loadmat(path)['CSIamp']
        mat[np.isinf(mat)] = np.nan
        for i in range(10):
            col = mat[:, :, i]
            nan_mask = np.isnan(col)
            if nan_mask.any():
                col[nan_mask] = col[~nan_mask].mean()
        mat = (mat - mat.min()) / (mat.max() - mat.min())
        return mat

    # ── Dataset interface ──

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        gt_torch = torch.from_numpy(np.load(item['gt_path']))

        if self.data_unit == 'sequence':
            sample = {
                'modality': item['modality'],
                'scene': item['scene'], 'subject': item['subject'],
                'action': item['action'], 'output': gt_torch,
            }
            for mod in item['modality']:
                dp = item[mod + '_path']
                sample['input_' + mod] = (
                    self._read_dir(dp) if os.path.isdir(dp)
                    else np.load(dp + '.npy')
                )
        else:  # frame
            frame_idx = item['idx']
            sample = {
                'modality': item['modality'],
                'scene': item['scene'], 'subject': item['subject'],
                'action': item['action'], 'idx': frame_idx,
                'output': gt_torch[frame_idx],
            }
            for mod in item['modality']:
                fp = item[mod + '_path']
                if not os.path.isfile(fp):
                    raise ValueError(f"Not a file: {fp}")
                sample['input_' + mod] = self._read_frame(fp)
        return sample


# ─── Factory Functions ────────────────────────────────────────────────────────

def make_dataset(dataset_root, config):
    """Return (train_dataset, val_dataset) for a given dataset root and config."""
    database = MMFi_Database(dataset_root)
    cfg = decode_config(config)
    train_ds = MMFi_Dataset(database, config['data_unit'], **cfg['train_dataset'])
    val_ds = MMFi_Dataset(database, config['data_unit'], **cfg['val_dataset'])
    return train_ds, val_ds


def _collate_fn(batch):
    """Collate frames into batched tensors; pad variable-length modalities."""
    batch_data = {
        'modality': batch[0]['modality'],
        'scene':    [s['scene']   for s in batch],
        'subject':  [s['subject'] for s in batch],
        'action':   [s['action']  for s in batch],
        'idx':      [s['idx']     for s in batch] if 'idx' in batch[0] else None,
        'output':   torch.FloatTensor(np.array([np.array(s['output']) for s in batch])),
    }
    for mod in batch_data['modality']:
        if mod in ('mmwave', 'lidar'):
            tensors = [torch.Tensor(s['input_' + mod]) for s in batch]
            padded = torch.nn.utils.rnn.pad_sequence(tensors)
            batch_data['input_' + mod] = padded.permute(1, 0, 2)
        else:
            arr = np.array([np.array(s['input_' + mod]) for s in batch])
            batch_data['input_' + mod] = torch.FloatTensor(arr)
    return batch_data


def make_dataloader(dataset, is_training, generator, batch_size,
                    num_workers=0, pin_memory=False, collate_fn=None):
    """Wrap a dataset in a DataLoader with sensible defaults."""
    if collate_fn is None:
        collate_fn = _collate_fn
    extra = {}
    if num_workers > 0:
        extra['persistent_workers'] = True
        extra['prefetch_factor'] = 2
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=is_training,
        drop_last=is_training,
        generator=generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **extra,
    )
