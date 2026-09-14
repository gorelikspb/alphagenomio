import csv
import io
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import app as ag


class FakeDnaClient:
    SEQUENCE_LENGTH_1MB = 32

    class OutputType:
        DNASE = "DNASE"


class FakeMetadata:
    def to_dict(self, _orient="records"):
        return [
            {
                "name": "UBERON:0002048 DNase-seq",
                "strand": ".",
                "biosample_name": "lung",
                "biosample_type": "tissue",
                "biosample_life_stage": "adult",
                "ontology_curie": "UBERON:0002048",
                "data_source": "encode",
                "nonzero_mean": 0.4,
            }
        ]


class FakeModel:
    """In-process stand-in for AlphaGenome. Signal depends on sequence content."""

    def __init__(self):
        self.calls = []

    def predict_sequence(self, sequence, requested_outputs, ontology_terms):
        self.calls.append(
            {
                "method": "predict_sequence",
                "sequence": sequence,
                "requested_outputs": list(requested_outputs),
                "ontology_terms": list(ontology_terms),
            }
        )
        return self._output(sequence)

    def _output(self, sequence):
        n = len(sequence)
        track0 = np.zeros(n, dtype=float)
        for i, ch in enumerate(sequence):
            track0[i] = {"A": 0.1, "C": 0.9, "G": 0.7, "T": 0.3, "N": 0.0}[ch]
            if ch == "C":
                track0[i] += 0.2
        values = np.stack([track0, track0 * 0.5], axis=1)
        dnase = SimpleNamespace(
            values=values,
            metadata=FakeMetadata(),
            resolution=1,
            width=n,
        )
        return SimpleNamespace(dnase=dnase)


def _parse_csv(text: str):
    reader = csv.DictReader(io.StringIO(text))
    return list(reader), reader.fieldnames


class AppTests(unittest.TestCase):
    def setUp(self):
        os.environ.pop("ALPHA_GENOME_API_KEY", None)
        ag._dna_model = None
        ag.app.config["TESTING"] = True
        self.client = ag.app.test_client()
        self.fake_model = FakeModel()
        self.patches = [
            patch.object(ag, "dna_client", FakeDnaClient),
            patch.object(ag, "get_model", return_value=self.fake_model),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        os.environ.pop("ALPHA_GENOME_API_KEY", None)
        ag._dna_model = None

    def test_landing_has_tissue_examples_compare_no_csv_yet(self):
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn('name="tissue"', html)
        self.assertIn("Lung", html)
        self.assertIn("Liver", html)
        self.assertIn("Brain", html)
        self.assertIn("Insert example", html)
        self.assertIn("Compare ref vs mutant", html)
        self.assertIn("AlphaGenome API key", html)
        self.assertNotIn("Download CSV", html)

    def test_byok_hidden_when_server_env_key_is_set(self):
        os.environ["ALPHA_GENOME_API_KEY"] = "server-key-not-for-commit"
        html = self.client.get("/").get_data(as_text=True)
        self.assertNotIn("AlphaGenome API key", html)
        self.assertNotIn("server-key-not-for-commit", html)

    def test_compare_shows_delta_table_and_csv_download(self):
        resp = self.client.post(
            "/",
            data={
                "mode": "compare",
                "reference": "GATTACA",
                "mutant": "GATTACG",
                "tissue": "lung",
                "api_key": "user-secret-key",
            },
        )
        html = resp.get_data(as_text=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Download CSV", html)
        self.assertIn("window.__COMPARE_CSV__", html)
        self.assertNotRegex(html, r'onclick="downloadCsv\([^)]+", "')
        self.assertIn("Delta per track", html)
        self.assertNotIn("user-secret-key", html)
        self.assertEqual(len(self.fake_model.calls), 2)
        ontologies = {tuple(c["ontology_terms"]) for c in self.fake_model.calls}
        self.assertEqual(ontologies, {("UBERON:0002048",)})

        parsed, fields = _parse_csv(self._csv_from_html(html, "alphagenomio-compare.csv"))
        self.assertEqual(
            list(fields),
            [
                "tissue",
                "ontology",
                "length",
                "changed_positions",
                "track",
                "ref_mean",
                "mut_mean",
                "delta_mean",
                "ref_max",
                "mut_max",
                "delta_max",
                "ref_peak_pos",
                "mut_peak_pos",
                "peak_shift",
            ],
        )
        self.assertGreaterEqual(len(parsed), 1)
        row0 = parsed[0]
        self.assertEqual(row0["tissue"], "Lung")
        self.assertEqual(row0["ontology"], "UBERON:0002048")
        self.assertEqual(row0["length"], "7")
        self.assertEqual(row0["changed_positions"], "7")
        self.assertEqual(row0["track"], "0")
        self.assertNotEqual(row0["delta_mean"], "0.0")
        self.assertNotEqual(row0["delta_mean"], "0.0000")

    def test_compare_liver_uses_liver_ontology(self):
        self.client.post(
            "/",
            data={
                "mode": "compare",
                "reference": "GATTACA",
                "mutant": "GATTACG",
                "tissue": "liver",
            },
        )
        ontologies = {tuple(c["ontology_terms"]) for c in self.fake_model.calls}
        self.assertEqual(ontologies, {("UBERON:0001114",)})

    def test_single_run_csv_download(self):
        resp = self.client.post(
            "/",
            data={
                "mode": "single",
                "sequence": "GATTACA",
                "tissue": "brain",
            },
        )
        html = resp.get_data(as_text=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Download CSV", html)
        self.assertIn("alphagenomio-dnase-stats.csv", html)
        self.assertIn("Prediction summary", html)
        parsed, fields = _parse_csv(self._csv_from_html(html, "alphagenomio-dnase-stats.csv"))
        self.assertEqual(
            list(fields),
            ["tissue", "ontology", "length", "track", "min", "mean", "max", "peak_pos"],
        )
        self.assertEqual(parsed[0]["tissue"], "Brain")
        self.assertEqual(parsed[0]["ontology"], "UBERON:0000955")
        self.assertEqual(parsed[0]["length"], "7")
        self.assertEqual(len(self.fake_model.calls), 1)

    def test_compare_validation_empty_mutant(self):
        resp = self.client.post(
            "/",
            data={
                "mode": "compare",
                "reference": "GATTACA",
                "mutant": "",
                "tissue": "lung",
            },
        )
        html = resp.get_data(as_text=True)
        self.assertIn("Please paste a mutant", html)
        self.assertNotIn("Download CSV", html)
        self.assertEqual(self.fake_model.calls, [])

    def test_compare_length_mismatch(self):
        resp = self.client.post(
            "/",
            data={
                "mode": "compare",
                "reference": "GATTACA",
                "mutant": "GATTAC",
                "tissue": "lung",
            },
        )
        html = resp.get_data(as_text=True)
        self.assertIn("same length", html)
        self.assertNotIn("Download CSV", html)

    def test_csv_helpers_roundtrip(self):
        compare = {
            "tissue_label": "Lung",
            "tissue_ontology": "UBERON:0002048",
            "input_length": 7,
            "diff_positions": [7],
            "deltas": [
                {
                    "track_index": 0,
                    "ref_mean": 0.3,
                    "mut_mean": 0.4,
                    "delta_mean": 0.1,
                    "ref_max": 0.7,
                    "mut_max": 0.7,
                    "delta_max": 0.0,
                    "ref_peak_pos": 4,
                    "mut_peak_pos": 4,
                    "peak_shift": 0,
                }
            ],
        }
        text = ag._compare_deltas_csv(compare)
        parsed, _ = _parse_csv(text)
        self.assertEqual(parsed[0]["delta_mean"], "0.1")
        self.assertEqual(parsed[0]["changed_positions"], "7")

        single = {
            "tissue_label": "Liver",
            "tissue_ontology": "UBERON:0001114",
            "input_length": 7,
            "segment_stats": [
                {"track_index": 0, "min": 0.1, "mean": 0.2, "max": 0.3, "max_pos": 2}
            ],
        }
        parsed, _ = _parse_csv(ag._single_stats_csv(single))
        self.assertEqual(parsed[0]["tissue"], "Liver")
        self.assertEqual(parsed[0]["peak_pos"], "2")

    def _csv_from_html(self, html: str, filename: str) -> str:
        import json
        import re

        self.assertIn(f"downloadCsv('{filename}'", html)
        var_name = "__COMPARE_CSV__" if filename.startswith("alphagenomio-compare") else "__SINGLE_CSV__"
        match = re.search(rf"window\.{var_name} = (.*?);\s*$", html, flags=re.MULTILINE)
        self.assertIsNotNone(match, f"missing JS payload for {filename}")
        return json.loads(match.group(1))


if __name__ == "__main__":
    unittest.main()
