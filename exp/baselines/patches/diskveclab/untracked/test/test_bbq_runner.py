import csv
import importlib.util
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src/BBQ/run_calibrated_adcbq.py"
SPEC = importlib.util.spec_from_file_location("diskveclab_bbq_runner", MODULE_PATH)
assert SPEC and SPEC.loader
bbq = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bbq
SPEC.loader.exec_module(bbq)


class CalibratedAdcBqTest(unittest.TestCase):
    def test_fit_target_mean_recall_produces_query_dependent_counts(self):
        ids = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64)
        distances = np.array([[0.0, 1.0, 2.0, 3.0], [0.0, 0.5, 4.0, 5.0]], dtype=np.float32)
        gt = np.array([[0, 2], [4, 5]], dtype=np.int32)
        scales = np.ones(2, dtype=np.float32)

        result = bbq.fit_target_mean_recall(ids, distances, gt, scales, k=2, target=0.75)

        self.assertEqual(result["tau"], 0.0)
        self.assertEqual(result["calibration_mean_recall"], 0.75)
        self.assertEqual(result["candidate_count_mean"], 2.0)
        self.assertEqual(result["traversal_recall_ceiling"], 1.0)

    def test_unreachable_target_preserves_traversal_misses(self):
        ids = np.array([[0, 1], [2, 3]], dtype=np.int64)
        distances = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        gt = np.array([[0], [9]], dtype=np.int32)
        scales = np.ones(2, dtype=np.float32)

        with self.assertRaises(bbq.CalibrationError):
            bbq.fit_target_mean_recall(ids, distances, gt, scales, k=1, target=0.75)

    def test_candidate_loader_requires_dense_query_rank_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidates.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["query_id", "rank", "id", "distance"])
                writer.writeheader()
                for query_id in range(2):
                    for rank in range(3):
                        writer.writerow(
                            {
                                "query_id": query_id,
                                "rank": rank,
                                "id": query_id * 3 + rank,
                                "distance": float(rank),
                            }
                        )
            ids, distances = bbq.load_candidates(path, nq=2, ef=3)
            self.assertEqual(ids.shape, (2, 3))
            np.testing.assert_allclose(distances[1], [0.0, 1.0, 2.0])

    def test_matrix_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.fbin"
            with path.open("wb") as handle:
                handle.write(struct.pack("ii", 3, 8))
                np.zeros((3, 8), dtype=np.float32).tofile(handle)
            self.assertEqual(bbq.matrix_header(path), (3, 8))


if __name__ == "__main__":
    unittest.main()
