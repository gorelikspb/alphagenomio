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


class AppTests(unittest.TestCase):
    def setUp(self):
        os.environ.pop("ALPHA_GENOME_API_KEY", None)
        os.environ.pop("ALPHAGENOME_MOCK", None)
        ag._dna_model = None
        ag.app.config["TESTING"] = True
        ag.app.config["SECRET_KEY"] = "test"
        self.client = ag.app.test_client()
        self.fake_model = FakeModel()
        self.patches = [
            patch.object(ag, "dna_client", FakeDnaClient),
            patch.object(ag, "_target_sequence_length", return_value=32),
            patch.object(ag, "get_model", return_value=self.fake_model),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def test_get_form_has_reference_mutant_compare_and_disclaimer(self):
        rv = self.client.get("/")
        html = rv.data.decode("utf-8")
        self.assertEqual(rv.status_code, 200)
        self.assertIn('name="reference"', html)
        self.assertIn('name="mutant"', html)
        self.assertIn('name="action" value="compare"', html)
        self.assertIn('name="action" value="run"', html)
        self.assertIn("not a Google or DeepMind product", html)
        self.assertIn("non-commercial", html)
        self.assertIn('name="api_key"', html)
        self.assertIn("UBERON:0002048", html)
        self.assertIn("GATTACA", html)

    def test_api_key_field_hidden_when_server_env_key_set(self):
        with patch.dict(os.environ, {"ALPHA_GENOME_API_KEY": "server-key-not-for-users"}):
            rv = self.client.get("/")
        html = rv.data.decode("utf-8")
        self.assertNotIn('name="api_key"', html)
        self.assertNotIn("server-key-not-for-users", html)

    def test_single_sequence_run_still_works(self):
        rv = self.client.post(
            "/",
            data={"reference": "GATTACA", "action": "run", "api_key": "SECRETKEYVALUE"},
        )
        html = rv.data.decode("utf-8")
        self.assertEqual(rv.status_code, 200)
        self.assertIn("Prediction summary", html)
        self.assertIn("Segment stats", html)
        self.assertNotIn("SECRETKEYVALUE", html)
        self.assertEqual(len(self.fake_model.calls), 1)
        self.assertEqual(
            self.fake_model.calls[0]["ontology_terms"], ["UBERON:0002048"]
        )
        self.assertEqual(
            self.fake_model.calls[0]["requested_outputs"], [FakeDnaClient.OutputType.DNASE]
        )

    def test_legacy_sequence_field_still_runs(self):
        rv = self.client.post("/", data={"sequence": "ACGT", "action": "run"})
        html = rv.data.decode("utf-8")
        self.assertIn("Prediction summary", html)
        self.assertIn("Input length:</strong> 4 bp", html)

    def test_compare_produces_delta_table(self):
        rv = self.client.post(
            "/",
            data={
                "reference": "GATTACA",
                "mutant": "GACTACA",
                "action": "compare",
                "ontology_term": "UBERON:0002048",
            },
        )
        html = rv.data.decode("utf-8")
        self.assertEqual(rv.status_code, 200)
        self.assertIn("Reference vs mutant", html)
        self.assertIn("Δ (mut − ref)", html)
        self.assertIn("Peak position", html)
        self.assertIn("Download CSV", html)
        self.assertIn("overlayChart", html)
        self.assertEqual(len(self.fake_model.calls), 2)
        ontologies = {tuple(c["ontology_terms"]) for c in self.fake_model.calls}
        outputs = {tuple(c["requested_outputs"]) for c in self.fake_model.calls}
        self.assertEqual(ontologies, {("UBERON:0002048",)})
        self.assertEqual(outputs, {(FakeDnaClient.OutputType.DNASE,)})

        # Same SNP should change mean on track 0 (T 0.3 vs C 0.9+0.2).
        deltas = ag.compute_track0_deltas(
            ag._summarize_prediction(
                self.fake_model._output("GATTACA".center(32, "N")),
                "GATTACA",
                "GATTACA".center(32, "N"),
            ),
            ag._summarize_prediction(
                self.fake_model._output("GACTACA".center(32, "N")),
                "GACTACA",
                "GACTACA".center(32, "N"),
            ),
        )
        self.assertGreater(deltas["delta_mean"], 0)
        self.assertIsInstance(deltas["peak_shift"], int)

    def test_compare_rejects_missing_mutant(self):
        rv = self.client.post(
            "/",
            data={"reference": "GATTACA", "mutant": "", "action": "compare"},
        )
        html = rv.data.decode("utf-8")
        self.assertIn("mutant DNA sequence", html)
        self.assertEqual(self.fake_model.calls, [])

    def test_invalid_bases_rejected(self):
        rv = self.client.post(
            "/",
            data={"reference": "GATTACA!", "action": "run"},
        )
        self.assertIn("must contain only A, C, G, T", rv.data.decode("utf-8"))
        self.assertEqual(self.fake_model.calls, [])

    def test_whitespace_in_sequence_is_stripped(self):
        rv = self.client.post(
            "/",
            data={"reference": "GA TT\nACA", "action": "run"},
        )
        self.assertIn("Input length:</strong> 7 bp", rv.data.decode("utf-8"))

    def test_unknown_tissue_falls_back_to_lung(self):
        rv = self.client.post(
            "/",
            data={
                "reference": "ACGT",
                "action": "run",
                "ontology_term": "UBERON:9999999",
            },
        )
        self.assertEqual(rv.status_code, 200)
        self.assertEqual(
            self.fake_model.calls[0]["ontology_terms"], ["UBERON:0002048"]
        )

    def test_brain_tissue_is_passed_through_to_both_compare_runs(self):
        rv = self.client.post(
            "/",
            data={
                "reference": "AAAA",
                "mutant": "CCCC",
                "action": "compare",
                "ontology_term": "UBERON:0000955",
            },
        )
        self.assertEqual(rv.status_code, 200)
        self.assertEqual(len(self.fake_model.calls), 2)
        self.assertTrue(
            all(c["ontology_terms"] == ["UBERON:0000955"] for c in self.fake_model.calls)
        )
        self.assertIn("UBERON:0000955 — Brain", rv.data.decode("utf-8"))

    def test_track0_delta_math(self):
        ref = {
            "input_length": 4,
            "segment_stats": [{"track_index": 0, "min": 0.1, "mean": 1.0, "max": 2.0, "max_pos": 2}],
        }
        mut = {
            "input_length": 4,
            "segment_stats": [{"track_index": 0, "min": 0.2, "mean": 1.5, "max": 3.5, "max_pos": 4}],
        }
        deltas = ag.compute_track0_deltas(ref, mut)
        self.assertAlmostEqual(deltas["delta_mean"], 0.5)
        self.assertAlmostEqual(deltas["delta_max"], 1.5)
        self.assertEqual(deltas["peak_shift"], 2)
        self.assertFalse(deltas["length_mismatch"])

    def test_error_redacts_api_key(self):
        msg = ag._public_error(RuntimeError("bad key SECRETKEYVALUE in request"), "SECRETKEYVALUE")
        self.assertNotIn("SECRETKEYVALUE", msg)
        self.assertIn("[redacted]", msg)

    def test_chart_data_is_built_for_single_run(self):
        rv = self.client.post("/", data={"reference": "ACGTACGT", "action": "run"})
        html = rv.data.decode("utf-8")
        self.assertIn("positionChart", html)
        self.assertIn("chart.js", html.lower())


if __name__ == "__main__":
    unittest.main()
