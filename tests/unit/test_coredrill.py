"""The coredrill's build options and the Coredrill page's data, without a model."""
import base64
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from train import coredrills  # noqa: E402
from train.data import eng  # noqa: E402

TAXONOMY = eng.load_taxonomy(REPO / "api" / "taxonomy.py")
LEAVES = [n["name"] for n in TAXONOMY if n["level"] == 5][:6]


def build(**options):
    rng = np.random.default_rng(0)
    records, rows = [], []
    for k, leaf in enumerate(LEAVES):
        for side in (0, 1):                      # every territory is two clouds
            center = rng.normal(size=32)
            for i in range(40):
                records.append({"text": f"{leaf} {side} {i}", "category": leaf, "source": "s", "lang": "en",
                                "article_title": f"{leaf}-{side}"})
                rows.append(center + (0.4 + 0.3 * k) * rng.normal(size=32))
    E = np.array(rows, dtype=np.float32)
    with contextlib.redirect_stdout(io.StringIO()):
        return eng.build_coredrill(records, E / np.linalg.norm(E, axis=1, keepdims=True), TAXONOMY, **options)


class Build(unittest.TestCase):
    def test_default_is_unchanged(self):
        cd = build()
        self.assertNotIn("centers", cd["tree"][LEAVES[0]])
        self.assertNotIn("sigma_raw", cd["tree"][LEAVES[0]])
        self.assertEqual(cd["build"]["radius_pull"], 0.0)

    def test_radius_pull(self):
        raw = [build()["tree"][n]["sigma"] for n in LEAVES]
        pulled = build(radius_pull=1.0)
        self.assertEqual(len({pulled["tree"][n]["sigma"] for n in LEAVES}), 1, "all the same at 1")
        self.assertEqual([pulled["tree"][n]["sigma_raw"] for n in LEAVES], raw)
        half = [build(radius_pull=0.5)["tree"][n]["sigma"] for n in LEAVES]
        self.assertLess(np.ptp(half), np.ptp(raw))

    def test_several_centers(self):
        cd = build(centers_per=30, centers_max=4)
        stats = cd["tree"][LEAVES[0]]
        C = np.array(stats["centers"])
        self.assertGreaterEqual(len(C), 2)
        np.testing.assert_allclose(np.linalg.norm(C, axis=1), 1, atol=1e-5)
        x = C[0] / np.linalg.norm(C[0])
        scored = eng._score_node_analysis(LEAVES[0], stats, x.astype(np.float32))
        self.assertAlmostEqual(scored["score"], 1.0, places=4)          # the nearest center, not the mean
        self.assertLess(float(x @ np.array(stats["centroid"])), 0.99)
        self.assertEqual(json.loads(json.dumps(cd))["tree"][LEAVES[0]]["centers"], stats["centers"])

    def test_centers_cache_follows_the_list(self):
        a, b = {"centers": [[1.0, 0.0]]}, {"centers": [[0.0, 1.0]]}
        self.assertEqual(eng.node_centers(a)[0, 0], 1.0)
        self.assertEqual(eng.node_centers(b)[0, 1], 1.0)
        self.assertIsNone(eng.node_centers({}))


class Live(unittest.TestCase):
    def test_territories_move(self):
        paths = {n["name"]: eng.taxonomy_lineage_paths(n["name"], eng.taxonomy_parents_map(TAXONOMY), {})[0] for n in TAXONOMY}
        labels = [n for n, p in paths.items() if len(p) > 2 and p[1] == "Life & Biology"][:8]
        rng = np.random.default_rng(1)
        P0 = rng.normal(size=(len(labels), 16))
        P0 /= np.linalg.norm(P0, axis=1, keepdims=True)
        P1 = P0 + 0.5 * rng.normal(size=P0.shape)
        P1 /= np.linalg.norm(P1, axis=1, keepdims=True)
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            loop = "20260101-000000-fit"
            (runs / loop / "live").mkdir(parents=True)
            (runs / coredrills.LEDGER).write_text(json.dumps({"event": "start", "id": loop, "seq": 1, "kind": "fit",
                                                              "started": "2026-01-01T00:00:00"}) + "\n", encoding="utf-8")
            meta = {"labels": labels, "taxonomy_paths": paths, "steps": 10, "epochs": 1,
                    "protos0": base64.b64encode(P0.astype("<f2").tobytes()).decode()}
            (runs / loop / "live" / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            G = np.round(np.clip(P1 @ P1.T, -1, 1) * 127).astype(np.int8)
            (runs / loop / "live" / "frame.json").write_text(json.dumps({"epoch": 1, "step": 5, "protos_cos": {
                "n": len(labels), "q": base64.b64encode(G.tobytes()).decode()}}), encoding="utf-8")
            listed = coredrills.listing(runs)
            self.assertTrue(any(c["live"] and c["id"] == loop for c in listed if c.get("live")))
            out = coredrills.live(runs, loop)
            nodes = {n["name"]: n for n in out["nodes"]}
            self.assertEqual(set(nodes), set(paths))
            moved = [np.linalg.norm(np.subtract(nodes[n]["to"], nodes[n]["from"])) for n in labels]
            self.assertGreater(max(moved), 0.01)
            for n in labels:
                S = P1 @ P1.T
                S[labels.index(n), labels.index(n)] = -1
                self.assertAlmostEqual(nodes[n]["b"]["gap"], 1 - S[labels.index(n)].max(), delta=0.02)
            self.assertEqual(out["live"]["step"], 5)


if __name__ == "__main__":
    unittest.main()
