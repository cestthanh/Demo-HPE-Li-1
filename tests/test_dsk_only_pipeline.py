import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feeder.splits import split_eval_dataset_by_sequence  # noqa: E402
from model import DSKConv, DSKNetMMFI3D  # noqa: E402
from tools.evaluate import load_model  # noqa: E402
from utils.eval_3d import compute_3d_metrics  # noqa: E402


class AblationConfigTests(unittest.TestCase):
    def test_default_data_config_matches_hpe_li_3d(self):
        path = PROJECT_ROOT / "config" / "mmfi" / "config_p1s1.yaml"
        with open(path, encoding="utf-8") as stream:
            config = yaml.safe_load(stream)

        self.assertEqual(config["protocol"], "protocol1")
        self.assertEqual(config["split_to_use"], "random_split")
        self.assertEqual(config["random_split"]["ratio"], 0.8)
        self.assertEqual(config["random_split"]["random_seed"], 0)

    def test_model_config_excludes_transformer_hyperparameters(self):
        config = DSKNetMMFI3D().get_model_config()
        self.assertEqual(config["num_lay"], 128)
        self.assertEqual(config["hidden_reg"], 32)
        self.assertEqual(config["sk_m"], 3)
        self.assertEqual(config["sk_g"], 32)
        self.assertNotIn("transformer_layers", config)
        self.assertNotIn("transformer_heads", config)


class MetricTests(unittest.TestCase):
    def test_coordinate_std_uses_mm_once(self):
        gt = np.zeros((2, 17, 3), dtype=np.float64)
        gt[1, :, 0] = 2.0
        metrics = compute_3d_metrics(gt.copy(), gt)

        self.assertEqual(metrics["gt_coord_std_mm_by_name"]["x"], 1000.0)
        self.assertEqual(metrics["pred_coord_std_mm_by_name"]["x"], 1000.0)
        self.assertEqual(metrics["coord_std_ratio_by_name"]["x"], 1.0)


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
