"""The parts the corpus and the coredrill are built from: no model, no data files."""
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import build_corpus  # noqa: E402
import gather_territories  # noqa: E402
from train.data import concept, eng, record_keys, text_key, title_key  # noqa: E402

TAXONOMY = eng.load_taxonomy(REPO / "api" / "taxonomy.py")
NAMES = {n["name"] for n in TAXONOMY}


class Taxonomy(unittest.TestCase):
    def test_one_tree(self):
        self.assertEqual(len(NAMES), len(TAXONOMY), "a node is listed twice")
        self.assertEqual(len(NAMES), 251)
        parents = eng.taxonomy_parents_map(TAXONOMY)
        for node in TAXONOMY:
            ps = parents[node["name"]]
            self.assertEqual(len(ps), 0 if node["level"] == 0 else 1, node["name"])
            for p in ps:
                self.assertIn(p, NAMES)

    def test_mappings_name_real_nodes(self):
        self.assertEqual(eng.check_mapping_targets(TAXONOMY), [])


class Sources(unittest.TestCase):
    articles = gather_territories.read_articles(REPO / "scripts" / "territory_articles.txt")

    def test_territories_exist(self):
        self.assertEqual(set(self.articles) - NAMES, set())
        self.assertEqual(build_corpus.CONVERSATIONAL - NAMES, set())

    def test_an_article_serves_one_territory(self):
        titles = [title_key(t) for ts in self.articles.values() for t in ts]
        self.assertEqual(sorted({t for t in titles if titles.count(t) > 1}), [])

    def test_no_territory_takes_anothers_article(self):
        own = {title_key(n): n for n in NAMES}
        taken = [(n, t) for n, ts in self.articles.items() for t in ts if own.get(title_key(t), n) != n]
        self.assertEqual(taken, [])

    def test_no_articles_for_conversation(self):
        self.assertEqual(set(self.articles) & build_corpus.CONVERSATIONAL, set())

    def test_paragraphs_stop_at_the_references(self):
        body = "x" * 10 + " " + "A sentence about rain. " * 12
        text = f"{body}\n== See also ==\n{body}"
        self.assertEqual(len(gather_territories.paragraphs(text)), 1)
        self.assertEqual(gather_territories.paragraphs("{\\displaystyle x} " + body), [])

    def test_relabel_never_invents_a_node(self):
        for source in ("cfpb", "medical_chatdoctor_patient", "medical_chatdoctor_doctor", "wikipedia"):
            label, _ = build_corpus.relabel({"text": "My bank charged me twice and I want my money back, please help me with this.",
                                             "source": source, "category": "Medicine"})
            self.assertTrue(label is None or label in NAMES)


class OneLabelPerText(unittest.TestCase):
    def test_merge_keeps_a_text_once(self):
        rows = [{"text": "The family is a group of people.", "category": c, "source": "wikipedia"}
                for c in ("Family Life", "Family Discussion", "Family Planning Talk")]
        with tempfile.TemporaryDirectory() as d:
            out = eng.merge_and_save(rows, [], Path(d) / "sentences.json")
            self.assertEqual([r["category"] for r in out], ["Family Life"])
            self.assertEqual(len(json.loads((Path(d) / "sentences.json").read_text(encoding="utf-8"))), 1)

    def test_the_named_territory_keeps_the_article(self):
        first = {"Epistemology": ["Epistemology"], "Knowledge & Belief": ["Epistemology", "Belief"]}
        self.assertEqual(build_corpus.choose(sorted(first), "epistemology", {}, first, {"Epistemology": 2, "Knowledge & Belief": 3}),
                         "Epistemology")
        self.assertEqual(build_corpus.choose(sorted(first), "belief", {"belief": "Knowledge & Belief"}, first, {}), "Knowledge & Belief")

    def test_record_keys_tell_copies_apart(self):
        r = {"text": "Rain  falls.", "category": "Climate & Weather"}
        keys = record_keys([r, dict(r, text="rain falls."), dict(r, category="Meteorology")])
        self.assertEqual(len(set(keys)), 3)
        self.assertEqual(text_key("Rain  falls."), text_key("rain falls. "))

    def test_languages_of_an_article_are_one_concept(self):
        rows = [{"source": "wikipedia_main_article", "source_article": "Nature"}, {"source": "wikipedia", "article_title": "Nature"},
                {"source": "simple_wikipedia", "article_title": "simple:Nature"}]
        self.assertEqual({concept(r) for r in rows}, {"wiki:nature"})
        self.assertIsNone(concept({"source": "sgd", "conversation": "x"}))


class EvenCoredrill(unittest.TestCase):
    def test_even_sample(self):
        groups = np.array(["big"] * 500 + ["small"] * 5, dtype=object)
        got = eng._even_sample_indices(np.arange(505), groups, 60)
        self.assertEqual(len(got), 60)
        self.assertEqual(len(set(got.tolist())), 60)
        self.assertEqual(int((got >= 500).sum()), 5, "a small group keeps all it has")
        self.assertGreater(int(got[got < 500].max()), 400, "a big group is sampled along its whole length")
        self.assertEqual(len(eng._even_sample_indices(np.arange(20), groups[:20], 60)), 20)

    def test_build_uses_the_cap(self):
        rng = np.random.default_rng(0)
        leaf = [n["name"] for n in TAXONOMY if n["level"] == 5][:2]
        records = [{"text": str(i), "category": leaf[i % 10 == 0], "source": "s", "article_title": f"a{i % 7}", "lang": "en"}
                   for i in range(400)]
        E = rng.normal(size=(400, 768)).astype(np.float32)
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            cd = eng.build_coredrill(records, E, TAXONOMY, per_node_cap=40)
        self.assertEqual({cd["tree"][n]["support_n"] for n in leaf}, {40})
        self.assertEqual(cd["tree"][leaf[0]]["direct_n"], 360)

    def test_centroid_update_matches_a_rebuild(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(30, 16))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        stats = {}
        eng.update_centroid_stats(stats, X[:20])
        eng.update_centroid_stats(stats, X[20:])
        mean = X.mean(axis=0)
        np.testing.assert_allclose(stats["centroid"], mean / np.linalg.norm(mean), atol=1e-5)
        self.assertAlmostEqual(stats["sigma"], 1 - np.linalg.norm(mean), places=5)


class Docs(unittest.TestCase):
    def test_mermaid(self):
        """GitHub cannot draw a sequence diagram with a semicolon in a message."""
        for f in [REPO / "README.md", *(REPO / "docs").glob("*.md")]:
            for chart in re.findall(r"```mermaid\n(.*?)```", f.read_text(encoding="utf-8").replace("\r\n", "\n"), re.S):
                if chart.lstrip().startswith("sequenceDiagram"):
                    self.assertNotIn(";", chart, f.name)

    def test_no_em_dashes(self):
        for f in [REPO / "README.md", *(REPO / "docs").glob("training.md"), *(REPO / "train").glob("*.*"), *(REPO / "scripts").glob("*.*")]:
            self.assertNotIn("—", f.read_text(encoding="utf-8"), f.name) if f.suffix in (".md", ".html", ".txt") else None


if __name__ == "__main__":
    unittest.main()
