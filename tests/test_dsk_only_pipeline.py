import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feeder.splits import split_eval_dataset_by_sequence  # noqa: E402
from model import DSKConv, DSKNetMMFI3D  # noqa: E402
from tools.evaluate import load_model  # noqa: E402
from train import (  # noqa: E402
    EARLY_STOPPING_MIN_DELTA_MM,
    EARLY_STOPPING_PATIENCE,
    GRAD_CLIP_NORM,
    POSE_STD_EPS,
    _make_training_metadata,
    _update_early_stopping_state,
    denormalize_pose,
    make_criterion,
    make_optimizer,
    normalize_pose,
)
from utils.eval_3d import compute_3d_metrics  # noqa: E402


class AblationConfigTests(unittest.TestCase):
    def test_default_data_config_matches_hpe_li_3d(self):
        path = (
            PROJECT_ROOT
            / "config"
            / "mmfi"
            / "config_phase_c_demo_2_p1s1.yaml"
        )
        with open(path, encoding="utf-8") as stream:
            config = yaml.safe_load(stream)

        self.assertEqual(config["modality"], "wifi-csi")
        self.assertEqual(config["protocol"], "protocol1")
        self.assertEqual(config["data_unit"], "frame")
        self.assertEqual(config["split_to_use"], "random_split")
        self.assertEqual(config["random_split"]["ratio"], 0.8)
        self.assertEqual(config["random_split"]["random_seed"], 0)
        self.assertEqual(config["init_rand_seed"], 0)
        self.assertEqual(config["train_loader"]["batch_size"], 16)
        self.assertEqual(config["val_loader"]["batch_size"], 8)
        self.assertEqual(config["test_loader"]["batch_size"], 8)
        for loader_name in ("train_loader", "val_loader", "test_loader"):
            self.assertEqual(config[loader_name]["num_workers"], 4)
            self.assertTrue(config[loader_name]["pin_memory"])

    def test_model_config_excludes_transformer_hyperparameters(self):
        config = DSKNetMMFI3D().get_model_config()
        self.assertEqual(config["num_lay"], 128)
        self.assertEqual(config["hidden_reg"], 32)
        self.assertEqual(config["sk_m"], 3)
        self.assertEqual(config["sk_g"], 32)
        self.assertNotIn("transformer_layers", config)
        self.assertNotIn("transformer_heads", config)


class MetricTests(unittest.TestCase):
    def test_core_metrics_only(self):
        gt = np.zeros((2, 17, 3), dtype=np.float64)
        metrics = compute_3d_metrics(gt.copy(), gt)

        self.assertEqual(metrics["mpjpe_mm"], 0.0)
        self.assertEqual(metrics["pck_50mm"], 100.0)
        self.assertNotIn("axis_mae_mm", metrics)
        self.assertNotIn("constant_mean_pose_mpjpe_mm", metrics)


class TrainingProfileTests(unittest.TestCase):
    def test_profile_is_fixed_to_normalized_mse_and_adam(self):
        criterion = make_criterion()
        optimizer_model = torch.nn.Linear(2, 1)
        optimizer = make_optimizer(optimizer_model, learning_rate=0.001)
        metadata = _make_training_metadata(
            SimpleNamespace(lr=0.001, epochs=60)
        )

        self.assertIsInstance(criterion, torch.nn.MSELoss)
        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertEqual(POSE_STD_EPS, 1e-6)
        self.assertEqual(GRAD_CLIP_NORM, 1.0)
        self.assertEqual(EARLY_STOPPING_PATIENCE, 15)
        self.assertEqual(EARLY_STOPPING_MIN_DELTA_MM, 0.2)
        self.assertEqual(metadata["loss"], "mse")
        self.assertEqual(metadata["pose_target_space"], "normalized_xyz")
        self.assertEqual(metadata["weight_decay"], 0.0)
        self.assertIsNone(metadata["lr_scheduler"])
        self.assertEqual(metadata["maximum_epochs"], 60)
        self.assertEqual(metadata["early_stopping_patience"], 15)
        self.assertEqual(metadata["early_stopping_min_delta_mm"], 0.2)
        self.assertEqual(metadata["checkpoint_selection_metric"], "val_mpjpe_mm")
        self.assertEqual(metadata["checkpoint_selection_mode"], "min")
        self.assertFalse(metadata["test_during_training"])
        self.assertFalse(metadata["transformer_enabled"])
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.0)

    def test_early_stopping_matches_reference_min_delta_semantics(self):
        best, count, is_best = _update_early_stopping_state(
            170.0, float("inf"), 0
        )
        self.assertEqual((best, count, is_best), (170.0, 0, True))

        best, count, is_best = _update_early_stopping_state(169.9, best, count)
        self.assertEqual((best, count, is_best), (169.9, 1, True))

        best, count, is_best = _update_early_stopping_state(170.1, best, count)
        self.assertEqual((best, count, is_best), (169.9, 2, False))

        best, count, is_best = _update_early_stopping_state(169.6, best, count)
        self.assertEqual((best, count, is_best), (169.6, 0, True))

    def test_pose_is_zscore_normalized_and_reconstructed(self):
        pose = torch.tensor([[[1.5, 4.0, 7.0]]])
        stats = {
            "mean": torch.tensor([1.0, 2.0, 3.0]).view(1, 1, 3),
            "std": torch.tensor([0.5, 2.0, 4.0]).view(1, 1, 3),
        }
        normalized = normalize_pose(pose, stats)

        torch.testing.assert_close(normalized, torch.ones_like(normalized))
        torch.testing.assert_close(denormalize_pose(normalized, stats), pose)


class SplitTests(unittest.TestCase):
    def test_sequence_split_has_no_overlap(self):
        class FakeDataset:
            def __init__(self):
                self.data_list = []
                for action in ("A01", "A02"):
                    for sequence_index in range(4):
                        gt_path = f"{action}/sequence_{sequence_index}/ground_truth.npy"
                        for frame_index in range(3):
                            self.data_list.append(
                                {
                                    "action": action,
                                    "gt_path": gt_path,
                                    "idx": frame_index,
                                }
                            )

            def __len__(self):
                return len(self.data_list)

        dataset = FakeDataset()
        val_dataset, test_dataset, metadata = split_eval_dataset_by_sequence(
            dataset,
            test_size=0.5,
            random_state=41,
        )
        val_paths = {
            dataset.data_list[index]["gt_path"] for index in val_dataset.indices
        }
        test_paths = {
            dataset.data_list[index]["gt_path"] for index in test_dataset.indices
        }

        self.assertFalse(val_paths & test_paths)
        self.assertEqual(metadata["stratify_key"], "action")
        self.assertEqual(metadata["sequence_overlap_count"], 0)


class DSKOnlyModelTests(unittest.TestCase):
    def test_dsk_block_preserves_input_shape(self):
        block = DSKConv(features=32, m=3, groups=8).eval()
        sample = torch.rand(2, 32, 9, 5)
        with torch.no_grad():
            output = block(sample)
        self.assertEqual(tuple(output.shape), tuple(sample.shape))

    def test_forward_shape_and_checkpoint_round_trip(self):
        torch.manual_seed(0)
        model = DSKNetMMFI3D().eval()
        restored = DSKNetMMFI3D(**model.get_model_config()).eval()
        restored.load_state_dict(model.state_dict())
        sample = torch.rand(1, 3, 114, 10)

        with torch.no_grad():
            expected, _ = model(sample)
            actual, _ = restored(sample)
            features = model._extract_features(sample)

        self.assertEqual(tuple(actual.shape), (1, 17, 3))
        self.assertEqual(tuple(features.shape), (1, 256, 14, 1))
        torch.testing.assert_close(actual, expected)

    def test_model_contains_no_transformer_modules_or_parameters(self):
        model = DSKNetMMFI3D()
        module_names = [name.lower() for name, _ in model.named_modules()]
        state_keys = [key.lower() for key in model.state_dict()]

        self.assertEqual(sum(param.numel() for param in model.parameters()), 393_363)
        self.assertFalse(any("transformer" in name for name in module_names))
        self.assertFalse(any("transformer" in key for key in state_keys))
        self.assertFalse(any("attention" in type(module).__name__.lower()
                             for module in model.modules()))

    def test_evaluator_loads_current_checkpoint(self):
        original = DSKNetMMFI3D().eval()
        checkpoint = {
            "model_name": "DSKNetMMFI3D",
            "model_config": original.get_model_config(),
            "model_state_dict": {
                f"module.{key}": value.clone()
                for key, value in original.state_dict().items()
            },
        }
        restored = load_model(checkpoint, torch.device("cpu"))
        sample = torch.rand(1, 3, 114, 10)

        with torch.no_grad():
            expected, _ = original(sample)
            actual, _ = restored(sample)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
