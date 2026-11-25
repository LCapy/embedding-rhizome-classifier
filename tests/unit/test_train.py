"""The training loop's bookkeeping and the classifier, without PyTorch or a model."""
import io
import pickle
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from train import encoder_np, forever, record  # noqa: E402
from train.classify import layout, universe  # noqa: E402
from train.data import eng, soft_target_matrix  # noqa: E402
from train.evaluate import paired_test  # noqa: E402


class Weights(unittest.TestCase):
    def test_only_tensors_are_read(self):
        class Evil:
            def __reduce__(self):
                return (print, ("this must not run",))

        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "weights.pt"
            with zipfile.ZipFile(f, "w") as z:
                z.writestr("weights/data.pkl", pickle.dumps({"encoder_delta": Evil()}))
            with self.assertRaises(pickle.UnpicklingError):
                encoder_np.read_weights(f)

    def test_storages_are_checked(self):
        u = encoder_np._Unpickler(b"", lambda key, count: np.zeros(count, dtype=np.float32))
        with self.assertRaises(pickle.UnpicklingError):
            u.persistent_load(("storage", "LongStorage", "0", "cpu", 4))
        with self.assertRaises(pickle.UnpicklingError):
            u.find_class("os", "system")
        np.testing.assert_array_equal(u.tensor(np.arange(6.0), 0, (2, 3), (3, 1)), np.arange(6.0).reshape(2, 3))
        with self.assertRaises(pickle.UnpicklingError):
            u.tensor(np.arange(6.0), 0, (2, 3), (1, 2))          # a transposed view

    def test_layer_norm(self):
        x = np.random.default_rng(0).normal(size=(3, 8))
        y = encoder_np._norm(x, np.ones(8), np.zeros(8), 1e-12)
        np.testing.assert_allclose(y.mean(axis=1), 0, atol=1e-12)
        np.testing.assert_allclose(y.std(axis=1), 1, atol=1e-6)


class Map(unittest.TestCase):
    def test_layout_recovers_three_dimensions(self):
        X = np.random.default_rng(0).normal(size=(40, 3))
        d = np.linalg.norm(X[:, None] - X[None], axis=-1)
        Y = layout(d)
        np.testing.assert_allclose(np.linalg.norm(Y[:, None] - Y[None], axis=-1), d, atol=0.05)

    def test_universe_keeps_domains_together(self):
        taxonomy = eng.load_taxonomy(REPO / "api" / "taxonomy.py")
        parents = eng.taxonomy_parents_map(taxonomy)
        memo = {}
        paths = {n["name"]: eng.taxonomy_lineage_paths(n["name"], parents, memo)[0] for n in taxonomy}
        rng = np.random.default_rng(0)
        tree = {n["name"]: {"centroid": (lambda v: (v / np.linalg.norm(v)).tolist())(rng.normal(size=64)), "level": n["level"]}
                for n in taxonomy}
        at = universe(tree, paths)
        self.assertEqual(set(at), set(tree))
        names = [n for n in at if len(paths[n]) > 1]
        P = np.array([at[n] for n in names])
        self.assertTrue(np.isfinite(P).all())
        d = np.linalg.norm(P[:, None] - P[None], axis=-1)
        np.fill_diagonal(d, np.inf)
        same = np.mean([paths[names[int(j)]][1] == paths[n][1] for n, j in zip(names, d.argmin(axis=1))])
        self.assertGreater(same, 0.9)


class Bookkeeping(unittest.TestCase):
    def test_paired_test(self):
        old = [{"key": str(i), "truth": "a", "deployed": "a" if i < 50 else "b"} for i in range(100)]
        new = [{"key": str(i), "truth": "a", "deployed": "a" if 10 <= i < 80 else "b"} for i in range(100)]
        t = paired_test(new, old)
        self.assertEqual((t["gained"], t["lost"]), (30, 10))
        self.assertAlmostEqual(t["z"], 20 / 40 ** 0.5, places=3)

    def test_soft_targets(self):
        labels = ["Branch", "Domain", "Topic"]
        T = soft_target_matrix(labels, {"Topic": ["Domain"], "Domain": ["Branch"], "Branch": []}, 0.1)
        np.testing.assert_allclose(T.sum(axis=1), 1, atol=1e-6)
        self.assertAlmostEqual(float(T[2, 2]), 0.9, places=6)
        self.assertGreater(T[2, 1], T[2, 0])

    def test_a_loop_whose_number_is_free_is_killed(self):
        self.assertEqual(record.state({"status": "did_not_finish", "pid": 99999996, "started": "2026-01-01T00:00:00"}), "killed")
        self.assertEqual(record.state({"status": "promoted", "pid": 1}), "promoted")

    def test_search_width_resumes(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "forever.log"
            self.assertEqual(forever.last_width(f), 0)
            f.write_text("x starting a round (search width 3 of 4, m)\nx loop 5: not_promoted\n", encoding="utf-8")
            self.assertEqual(forever.last_width(f), 2)
            f.write_text(f.read_text(encoding="utf-8") + "x loop 6: promoted, deployed\n", encoding="utf-8")
            self.assertEqual(forever.last_width(f), 0)

    def test_nothing_here_loads_pytorch(self):
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
