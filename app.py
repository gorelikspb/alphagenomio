import csv
import io
import json
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from flask import Flask, render_template, request

load_dotenv()

try:
    from alphagenome.models import dna_client
except ImportError:
    dna_client = None  # type: ignore


app = Flask(__name__)

# Lung, as in the AlphaGenome quick-start example. Keep as the default.
DEFAULT_ONTOLOGY = "UBERON:0002048"
TISSUE_OPTIONS: List[Tuple[str, str]] = [
    ("UBERON:0002048", "Lung"),
    ("UBERON:0000955", "Brain"),
    ("UBERON:0000948", "Heart"),
    ("UBERON:0001114", "Liver"),
    ("UBERON:0002113", "Kidney"),
    ("UBERON:0001155", "Colon"),
]
ALLOWED_ONTOLOGY_TERMS = {term for term, _label in TISSUE_OPTIONS}

SEQUENCE_PRESETS: List[Dict[str, str]] = [
    {
        "id": "gattaca",
        "label": "GATTACA SNP",
        "reference": "GATTACA",
        "mutant": "GACTACA",
    },
    {
        "id": "gc_snp",
        "label": "60 bp GC-rich SNP",
        "reference": "GCGCCCGCCTATAATAGCGCGCGCATTGCGCGCGCGCGCGCGCGATTTAAAGCGCGCG",
        "mutant": "GCGCCCGCCTATAATAGCGCGCGCCTTGGCGCGCGCGCGCGCGCGATTTAAAGCGCGCG",
    },
]

_VALID_BASES = set("ACGTN")


class _FallbackDna:
    """Used only when ALPHAGENOME_MOCK=1 and the real client is not installed."""

    SEQUENCE_LENGTH_1MB = 1_048_576

    class OutputType:
        DNASE = "DNASE"


def _require_client():
    if dna_client is not None:
        return dna_client
    if os.environ.get("ALPHAGENOME_MOCK") == "1":
        return _FallbackDna
    raise RuntimeError(
        "alphagenome is not installed. Run `pip install -r requirements.txt` first."
    )


def _target_sequence_length() -> int:
    return int(_require_client().SEQUENCE_LENGTH_1MB)


def _ontology_label(term: str) -> str:
    for curie, label in TISSUE_OPTIONS:
        if curie == term:
            return label
    return term


def _resolve_ontology(raw: Optional[str]) -> str:
    term = (raw or "").strip()
    if term in ALLOWED_ONTOLOGY_TERMS:
        return term
    return DEFAULT_ONTOLOGY


def _normalize_dna(raw: Optional[str]) -> str:
    """Uppercase and drop whitespace so pasted sequences with line breaks work."""
    return "".join((raw or "").split()).upper()


def _validate_sequence(sequence: str, field_name: str) -> Optional[str]:
    if not sequence:
        return f"Please paste a {field_name} DNA sequence."
    if any(ch not in _VALID_BASES for ch in sequence):
        return f"{field_name} sequence must contain only A, C, G, T (and optionally N)."
    return None


def _public_error(exc: BaseException, api_key: Optional[str]) -> str:
    """Surface the error without echoing any API key that might appear in a message."""
    msg = str(exc)
    for secret in (api_key, os.environ.get("ALPHA_GENOME_API_KEY")):
        if secret:
            msg = msg.replace(secret, "[redacted]")
    return f"Error while calling AlphaGenome: {msg}"


def create_model(api_key: Optional[str] = None):
    """Create AlphaGenome DNA client. Uses api_key if given, else ALPHA_GENOME_API_KEY env."""
    if os.environ.get("ALPHAGENOME_MOCK") == "1":
        return _MockDnaModel()
    if dna_client is None:
        raise RuntimeError(
            "alphagenome is not installed. Run `pip install -r requirements.txt` first."
        )
    key = api_key or os.environ.get("ALPHA_GENOME_API_KEY")
    if not key:
        raise RuntimeError(
            "No API key provided. Enter it in the field, or set ALPHA_GENOME_API_KEY on the server."
        )
    return dna_client.create(key)


_dna_model = None


def get_model(api_key: Optional[str] = None):
    """Cached model when using env key; fresh model when api_key is provided (e.g. from form)."""
    if api_key:
        return create_model(api_key)
    global _dna_model
    if _dna_model is None:
        _dna_model = create_model()
    return _dna_model


class _MockMetadata:
    def __init__(self, ontology_term: str):
        self._ontology_term = ontology_term

    def to_dict(self, _orient: str = "records") -> List[dict]:
        label = _ontology_label(self._ontology_term)
        return [
            {
                "name": f"{self._ontology_term} DNase-seq",
                "strand": ".",
                "biosample_name": label.lower(),
                "biosample_type": "tissue",
                "biosample_life_stage": "adult",
                "ontology_curie": self._ontology_term,
                "data_source": "mock",
                "nonzero_mean": 0.4,
            },
            {
                "name": f"{self._ontology_term} DNase-seq alt",
                "strand": ".",
                "biosample_name": label.lower(),
                "biosample_type": "tissue",
                "biosample_life_stage": "adult",
                "ontology_curie": self._ontology_term,
                "data_source": "mock",
                "nonzero_mean": 0.2,
            },
        ]


class _MockDnaModel:
    """Deterministic stand-in for local UI tests. Never used on the live BYOK path."""

    def predict_sequence(self, sequence, requested_outputs, ontology_terms, **_kwargs):
        return self._build_output(sequence, ontology_terms)

    def predict_sequences(self, sequences, requested_outputs, ontology_terms, **_kwargs):
        return [self._build_output(seq, ontology_terms) for seq in sequences]

    def _build_output(self, sequence: str, ontology_terms: Sequence[str]):
        import numpy as np

        seq = sequence or ""
        n = len(seq)
        codes = np.frombuffer(seq.encode("ascii"), dtype=np.uint8) if n else np.zeros(0, dtype=np.uint8)
        # A,C,G,T,N get distinct weights so a SNP changes track 0.
        weights = np.zeros(n, dtype=float)
        weights[codes == ord("A")] = 0.15
        weights[codes == ord("C")] = 0.95
        weights[codes == ord("G")] = 0.80
        weights[codes == ord("T")] = 0.25
        idx = np.arange(n, dtype=float)
        bump = 0.15 * np.sin((idx / max(n, 1)) * 6.0) ** 2
        track0 = weights + bump
        track1 = track0 * 0.45
        values = np.stack([track0, track1], axis=1) if n else np.zeros((0, 2))
        term = ontology_terms[0] if ontology_terms else DEFAULT_ONTOLOGY
        dnase = SimpleNamespace(
            values=values,
            metadata=_MockMetadata(term),
            resolution=1,
            width=n,
        )
        return SimpleNamespace(dnase=dnase)


def _to_serializable(obj: Any) -> Any:
    """Convert numpy/pandas types to native Python for JSON display."""
    try:
        import numpy as np
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


def _build_api_raw_output(output, segment, start_idx: int, seq_len: int) -> dict:
    """Build a serializable view of the full API response for display."""
    raw: dict = {}
    if output.dnase is None:
        raw["dnase"] = None
        return raw
    dnase = output.dnase
    vals = dnase.values
    try:
        raw["dnase"] = {
            "values_shape": list(vals.shape),
            "resolution": getattr(dnase, "resolution", None),
            "width": getattr(dnase, "width", None),
        }
    except Exception:
        raw["dnase"] = {"values_shape": "unknown"}
        return raw
    # Metadata (track names, strand, etc.)
    if hasattr(dnase, "metadata") and dnase.metadata is not None:
        try:
            meta = dnase.metadata
            if hasattr(meta, "to_dict"):
                records = meta.to_dict("records")
            else:
                records = [dict(meta)]
            clean = []
            for row in records:
                clean.append({k: _to_serializable(v) for k, v in row.items()})
            raw["dnase"]["metadata"] = clean
        except Exception:
            raw["dnase"]["metadata"] = None
    else:
        raw["dnase"]["metadata"] = None
    # Sample of values over the user's sequence (first 30 positions, all tracks)
    try:
        import numpy as np
        seg = np.asarray(segment)
        n_pos = min(30, seg.shape[0])
        n_tr = min(10, seg.shape[1] if seg.ndim > 1 else 1)
        if seg.ndim == 1:
            sample = [[round(float(seg[i]), 4)] for i in range(n_pos)]
        else:
            sample = [[round(float(seg[i, j]), 4) for j in range(n_tr)] for i in range(n_pos)]
        raw["dnase"]["segment_sample"] = {
            "description": f"First {n_pos} positions × first {n_tr} tracks (your sequence segment)",
            "data": sample,
        }
    except Exception:
        raw["dnase"]["segment_sample"] = None
    return raw


def _summarize_segment_stats(segment, max_tracks: int = 10) -> List[dict]:
    """Compute compact per-track stats over the user's segment."""
    try:
        import numpy as np
    except ImportError:
        # Minimal fallback without numpy
        stats: List[dict] = []
        num_pos = len(segment)
        num_tracks = len(segment[0]) if num_pos and hasattr(segment[0], "__len__") else 1
        num_tracks = min(num_tracks, max_tracks)
        for j in range(num_tracks):
            vals = [float(segment[i][j]) for i in range(num_pos)]
            vmin = min(vals) if vals else 0.0
            vmax = max(vals) if vals else 0.0
            mean = sum(vals) / max(len(vals), 1)
            argmax = vals.index(vmax) + 1 if vals else None
            stats.append({"track_index": j, "min": vmin, "mean": mean, "max": vmax, "max_pos": argmax})
        return stats

    seg = np.asarray(segment, dtype=float)
    if seg.ndim == 1:
        seg = seg.reshape(-1, 1)
    seg = seg[:, :max_tracks]
    mins = seg.min(axis=0)
    maxs = seg.max(axis=0)
    means = seg.mean(axis=0)
    argmax = seg.argmax(axis=0) + 1  # 1-based position within segment
    out: List[dict] = []
    for j in range(seg.shape[1]):
        out.append(
            {
                "track_index": int(j),
                "min": float(mins[j]),
                "mean": float(means[j]),
                "max": float(maxs[j]),
                "max_pos": int(argmax[j]),
            }
        )
    return out


def _compact_track_metadata(output, max_tracks: int = 10) -> Optional[List[dict]]:
    """Return a compact view of TrackData.metadata for quick UI display."""
    if output.dnase is None or not hasattr(output.dnase, "metadata") or output.dnase.metadata is None:
        return None
    meta = output.dnase.metadata
    try:
        records = meta.to_dict("records")
    except Exception:
        return None

    keep_cols = [
        "name",
        "strand",
        "biosample_name",
        "biosample_type",
        "biosample_life_stage",
        "ontology_curie",
        "data_source",
        "nonzero_mean",
    ]
    compact: List[dict] = []
    for row in records[:max_tracks]:
        compact.append({k: _to_serializable(row.get(k)) for k in keep_cols if k in row})
    return compact


def _top_peaks(segment, k: int = 5, track_index: int = 0) -> List[dict]:
    """Return top-k peak positions within the user's segment for a given track."""
    try:
        import numpy as np
    except ImportError:
        vals = [float(segment[i][track_index]) for i in range(len(segment))] if segment else []
        ranked = sorted(enumerate(vals, start=1), key=lambda t: t[1], reverse=True)[:k]
        return [{"pos": int(pos), "value": float(v)} for pos, v in ranked]

    seg = np.asarray(segment, dtype=float)
    if seg.ndim == 1:
        seg = seg.reshape(-1, 1)
    if seg.shape[0] == 0:
        return []
    j = min(track_index, seg.shape[1] - 1)
    v = seg[:, j]
    k = min(k, v.shape[0])
    # argpartition for speed, then sort those indices by value desc
    idx = np.argpartition(-v, k - 1)[:k]
    idx = idx[np.argsort(-v[idx])]
    return [{"pos": int(i + 1), "value": float(v[i])} for i in idx]  # 1-based


def _build_chart_data(segment, sequence: str, max_positions: int = 300, max_tracks: int = 5) -> Optional[dict]:
    """Build downsampled position-level data for Chart.js visualisation."""
    try:
        import numpy as np
    except ImportError:
        return None

    seg = np.asarray(segment, dtype=float)
    if seg.ndim == 1:
        seg = seg.reshape(-1, 1)
    n_pos, n_tracks = seg.shape
    if n_pos == 0:
        return None

    n_tracks = min(n_tracks, max_tracks)
    seg = seg[:, :n_tracks]

    if n_pos > max_positions:
        step = n_pos / max_positions
        indices = np.round(np.arange(0, n_pos, step)).astype(int)[:max_positions]
        seg = seg[indices]
        labels = [f"{sequence[i]}{i + 1}" if i < len(sequence) else str(i + 1) for i in indices]
    else:
        labels = [
            sequence[i] + str(i + 1) if i < len(sequence) else str(i + 1)
            for i in range(n_pos)
        ]

    datasets = []
    for j in range(n_tracks):
        datasets.append({
            "label": f"Track {j}",
            "data": [round(float(v), 5) for v in seg[:, j]],
        })

    return {"labels": labels, "datasets": datasets}


def _overlay_chart_data(ref_chart: Optional[dict], mut_chart: Optional[dict]) -> Optional[dict]:
    """Track-0 overlay for reference vs mutant. Pads the shorter series with nulls."""
    if not ref_chart or not mut_chart:
        return None
    if not ref_chart.get("datasets") or not mut_chart.get("datasets"):
        return None
    ref_vals = list(ref_chart["datasets"][0].get("data") or [])
    mut_vals = list(mut_chart["datasets"][0].get("data") or [])
    n = max(len(ref_vals), len(mut_vals))
    if n == 0:
        return None
    labels = [str(i + 1) for i in range(n)]
    ref_vals = ref_vals + [None] * (n - len(ref_vals))
    mut_vals = mut_vals + [None] * (n - len(mut_vals))
    return {
        "labels": labels,
        "datasets": [
            {"label": "Reference (track 0)", "data": ref_vals},
            {"label": "Mutant (track 0)", "data": mut_vals},
        ],
    }


def _highlighted_sequence(sequence: str, peak_positions_1based: List[int]) -> List[dict]:
    """Return a per-base structure for templating with peak highlights."""
    peak_set = set(int(p) for p in peak_positions_1based)
    out: List[dict] = []
    for i, ch in enumerate(sequence, start=1):
        out.append({"pos": i, "ch": ch, "is_peak": i in peak_set})
    return out


def _sequence_mismatches(reference: str, mutant: str) -> Optional[List[dict]]:
    if len(reference) != len(mutant):
        return None
    diffs: List[dict] = []
    for i, (a, b) in enumerate(zip(reference, mutant), start=1):
        if a != b:
            diffs.append({"pos": i, "ref": a, "mut": b})
    return diffs


def summarize_dnase_predictions(values) -> List[float]:
    """
    Take a 2D array (sequence_length x num_tracks) and return
    mean value per track as a small list of floats.
    """
    try:
        import numpy as np
    except ImportError:
        # Fallback without numpy (very slow for large arrays, but ok for demo)
        num_positions = len(values)
        num_tracks = len(values[0]) if num_positions else 0
        means: List[float] = []
        for j in range(num_tracks):
            s = 0.0
            for i in range(num_positions):
                s += float(values[i][j])
            means.append(s / max(num_positions, 1))
        return means

    arr = np.asarray(values, dtype=float)
    # mean over positions (axis 0), keep tracks
    return arr.mean(axis=0).tolist()


def _pad_sequence(sequence: str) -> str:
    target = _target_sequence_length()
    if len(sequence) > target:
        raise ValueError(
            f"Sequence is longer than the {target} bp AlphaGenome context window."
        )
    return sequence.center(target, "N")


def _json_dump(payload: dict) -> str:
    try:
        return json.dumps(payload, indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        return '{"note": "Full API response could not be serialized to JSON."}'


def _summarize_prediction(output, sequence: str, padded: str) -> dict:
    if output.dnase is None:
        raise RuntimeError("AlphaGenome returned no DNase output for this request.")
    dnase_values = output.dnase.values  # shape: (padded_length, num_tracks)
    start_idx = (len(padded) - len(sequence)) // 2
    segment = dnase_values[start_idx : start_idx + len(sequence)]
    means = summarize_dnase_predictions(segment)
    segment_stats = _summarize_segment_stats(segment, max_tracks=10)
    track_meta = _compact_track_metadata(output, max_tracks=10)
    peaks_t0 = _top_peaks(segment, k=5, track_index=0)
    highlighted_seq = _highlighted_sequence(sequence, [p["pos"] for p in peaks_t0])
    chart_data = _build_chart_data(segment, sequence)
    api_raw = _build_api_raw_output(output, segment, start_idx, len(sequence))
    return {
        "input_length": len(sequence),
        "padded_length": len(padded),
        "num_tracks": len(means),
        "track_means": [round(m, 4) for m in means[:10]],
        "segment_stats": segment_stats,
        "track_meta": track_meta,
        "peaks_t0": peaks_t0,
        "highlighted_seq": highlighted_seq,
        "chart_data": chart_data,
        "api_raw_json": _json_dump(api_raw),
    }


def _predict_outputs(model, sequences: Sequence[str], ontology_term: str):
    """Run DNase predictions with the same tissue/output config for every sequence."""
    client = _require_client()
    padded_list = [_pad_sequence(seq) for seq in sequences]
    kwargs = {
        "requested_outputs": [client.OutputType.DNASE],
        "ontology_terms": [ontology_term],
    }
    if len(sequences) == 1 or not hasattr(model, "predict_sequences"):
        outputs = [
            model.predict_sequence(sequence=padded, **kwargs) for padded in padded_list
        ]
    else:
        try:
            outputs = model.predict_sequences(
                sequences=padded_list, progress_bar=False, **kwargs
            )
        except TypeError:
            outputs = model.predict_sequences(sequences=padded_list, **kwargs)
    summarized = []
    for seq, padded, output in zip(sequences, padded_list, outputs):
        summarized.append(_summarize_prediction(output, seq, padded))
    return summarized


def _track0_stats(result: dict) -> dict:
    stats = result.get("segment_stats") or []
    if stats:
        return stats[0]
    return {"track_index": 0, "min": 0.0, "mean": 0.0, "max": 0.0, "max_pos": None}


def compute_track0_deltas(ref_result: dict, mut_result: dict) -> dict:
    """Δmean, Δmax, and peak shift on track 0 (mutant − reference)."""
    ref = _track0_stats(ref_result)
    mut = _track0_stats(mut_result)
    ref_peak = ref.get("max_pos")
    mut_peak = mut.get("max_pos")
    peak_shift = None
    if ref_peak is not None and mut_peak is not None:
        peak_shift = int(mut_peak) - int(ref_peak)
    return {
        "ref_mean": float(ref.get("mean") or 0.0),
        "mut_mean": float(mut.get("mean") or 0.0),
        "delta_mean": float(mut.get("mean") or 0.0) - float(ref.get("mean") or 0.0),
        "ref_max": float(ref.get("max") or 0.0),
        "mut_max": float(mut.get("max") or 0.0),
        "delta_max": float(mut.get("max") or 0.0) - float(ref.get("max") or 0.0),
        "ref_peak_pos": ref_peak,
        "mut_peak_pos": mut_peak,
        "peak_shift": peak_shift,
        "length_mismatch": ref_result.get("input_length") != mut_result.get("input_length"),
    }


def _csv_from_rows(headers: List[str], rows: List[List[Any]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(headers)
    writer.writerows(rows)
    return buf.getvalue()


def _single_csv(result: dict, ontology_term: str) -> str:
    rows: List[List[Any]] = []
    for st in result.get("segment_stats") or []:
        rows.append(
            [
                ontology_term,
                st.get("track_index"),
                round(float(st.get("min") or 0.0), 6),
                round(float(st.get("mean") or 0.0), 6),
                round(float(st.get("max") or 0.0), 6),
                st.get("max_pos"),
            ]
        )
    return _csv_from_rows(
        ["ontology", "track", "min", "mean", "max", "peak_pos"],
        rows,
    )


def _compare_csv(deltas: dict, ontology_term: str, mismatches: Optional[List[dict]]) -> str:
    rows = [
        [ontology_term, "track0_mean", deltas["ref_mean"], deltas["mut_mean"], deltas["delta_mean"]],
        [ontology_term, "track0_max", deltas["ref_max"], deltas["mut_max"], deltas["delta_max"]],
        [
            ontology_term,
            "track0_peak_pos",
            deltas["ref_peak_pos"],
            deltas["mut_peak_pos"],
            deltas["peak_shift"],
        ],
    ]
    text = _csv_from_rows(
        ["ontology", "metric", "reference", "mutant", "delta_mut_minus_ref"],
        rows,
    )
    if mismatches:
        extra = _csv_from_rows(
            ["mismatch_pos", "reference_base", "mutant_base"],
            [[d["pos"], d["ref"], d["mut"]] for d in mismatches],
        )
        text = text + "\n" + extra
    return text


def _form_context(
    *,
    reference: str = "",
    mutant: str = "",
    ontology_term: str = DEFAULT_ONTOLOGY,
    result: Optional[dict] = None,
    error: Optional[str] = None,
) -> dict:
    return {
        "reference": reference,
        "mutant": mutant,
        "ontology_term": ontology_term,
        "tissue_options": TISSUE_OPTIONS,
        "presets": SEQUENCE_PRESETS,
        "result": result,
        "error": error,
        "show_api_key_field": not bool(os.environ.get("ALPHA_GENOME_API_KEY")),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    reference = ""
    mutant = ""
    ontology_term = DEFAULT_ONTOLOGY
    result: Optional[dict] = None
    error: Optional[str] = None

    if request.method == "POST":
        # Accept legacy "sequence" so the previous single-field POST still runs.
        reference = _normalize_dna(
            request.form.get("reference") or request.form.get("sequence")
        )
        mutant = _normalize_dna(request.form.get("mutant"))
        ontology_term = _resolve_ontology(request.form.get("ontology_term"))
        action = (request.form.get("action") or "run").strip().lower()
        api_key = (request.form.get("api_key") or "").strip() or None

        if action == "compare":
            error = _validate_sequence(reference, "reference") or _validate_sequence(
                mutant, "mutant"
            )
        else:
            error = _validate_sequence(reference, "DNA")

        if error is None:
            try:
                model = get_model(api_key)
                if action == "compare":
                    ref_out, mut_out = _predict_outputs(
                        model, [reference, mutant], ontology_term
                    )
                    deltas = compute_track0_deltas(ref_out, mut_out)
                    mismatches = _sequence_mismatches(reference, mutant)
                    result = {
                        "mode": "compare",
                        "ontology_term": ontology_term,
                        "ontology_label": _ontology_label(ontology_term),
                        "reference": ref_out,
                        "mutant": mut_out,
                        "deltas_t0": deltas,
                        "mismatches": mismatches,
                        "overlay_chart": _overlay_chart_data(
                            ref_out.get("chart_data"), mut_out.get("chart_data")
                        ),
                        "csv_text": _compare_csv(deltas, ontology_term, mismatches),
                    }
                else:
                    single = _predict_outputs(model, [reference], ontology_term)[0]
                    result = {
                        "mode": "single",
                        "ontology_term": ontology_term,
                        "ontology_label": _ontology_label(ontology_term),
                        "csv_text": _single_csv(single, ontology_term),
                        **single,
                    }
            except Exception as exc:  # noqa: BLE001
                error = _public_error(exc, api_key)

    return render_template("index.html", **_form_context(
        reference=reference,
        mutant=mutant,
        ontology_term=ontology_term,
        result=result,
        error=error,
    ))


if __name__ == "__main__":
    # debug=True only for local experiments.
    app.run(host="127.0.0.1", port=5000, debug=True)
