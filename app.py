import csv
import io
import json
import os
from typing import Any, List, Optional

from dotenv import load_dotenv
from flask import Flask, render_template, request

load_dotenv()

try:
    from alphagenome.models import dna_client
except ImportError:
    dna_client = None  # type: ignore


app = Flask(__name__)

TISSUE_OPTIONS = {
    "lung": {"label": "Lung", "ontology": "UBERON:0002048"},
    "liver": {"label": "Liver", "ontology": "UBERON:0001114"},
    "brain": {"label": "Brain", "ontology": "UBERON:0000955"},
}
DEFAULT_TISSUE = "lung"


def create_model(api_key: Optional[str] = None):
    """Create AlphaGenome DNA client. Uses api_key if given, else ALPHA_GENOME_API_KEY env."""
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


def _highlighted_sequence(sequence: str, peak_positions_1based: List[int]) -> List[dict]:
    """Return a per-base structure for templating with peak highlights."""
    peak_set = set(int(p) for p in peak_positions_1based)
    out: List[dict] = []
    for i, ch in enumerate(sequence, start=1):
        out.append({"pos": i, "ch": ch, "is_peak": i in peak_set})
    return out


def _validate_sequence(seq: str, label: str = "Sequence") -> Optional[str]:
    if not seq:
        return f"Please paste a {label.lower()}."
    if any(ch not in "ACGTN" for ch in seq):
        return f"{label} must contain only A, C, G, T (and optionally N)."
    return None


def _predict_dnase(model, sequence: str, tissue: str = DEFAULT_TISSUE) -> dict:
    """Run DNase prediction for one sequence; return display-ready result fields."""
    tissue_info = TISSUE_OPTIONS.get(tissue, TISSUE_OPTIONS[DEFAULT_TISSUE])
    padded = sequence.center(
        dna_client.SEQUENCE_LENGTH_1MB, "N"  # type: ignore[attr-defined]
    )
    output = model.predict_sequence(
        sequence=padded,
        requested_outputs=[dna_client.OutputType.DNASE],  # type: ignore[attr-defined]
        ontology_terms=[tissue_info["ontology"]],
    )
    dnase_values = output.dnase.values
    start_idx = (len(padded) - len(sequence)) // 2
    segment = dnase_values[start_idx : start_idx + len(sequence)]
    means = summarize_dnase_predictions(segment)
    segment_stats = _summarize_segment_stats(segment, max_tracks=10)
    track_meta = _compact_track_metadata(output, max_tracks=10)
    peaks_t0 = _top_peaks(segment, k=5, track_index=0)
    highlighted_seq = _highlighted_sequence(sequence, [p["pos"] for p in peaks_t0])
    api_raw = _build_api_raw_output(output, segment, start_idx, len(sequence))
    try:
        api_raw_json = json.dumps(api_raw, indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        api_raw_json = '{"note": "Full API response could not be serialized to JSON."}'

    return {
        "sequence": sequence,
        "tissue_key": tissue if tissue in TISSUE_OPTIONS else DEFAULT_TISSUE,
        "tissue_label": tissue_info["label"],
        "tissue_ontology": tissue_info["ontology"],
        "input_length": len(sequence),
        "padded_length": len(padded),
        "num_tracks": len(means),
        "track_means": [round(m, 4) for m in means[:10]],
        "segment_stats": segment_stats,
        "track_meta": track_meta,
        "peaks_t0": peaks_t0,
        "highlighted_seq": highlighted_seq,
        "api_raw_json": api_raw_json,
    }


def _diff_positions(ref: str, mut: str) -> List[int]:
    return [i + 1 for i, (a, b) in enumerate(zip(ref, mut)) if a != b]


def _compare_stats(ref_stats: List[dict], mut_stats: List[dict]) -> List[dict]:
    deltas: List[dict] = []
    for ref_st, mut_st in zip(ref_stats, mut_stats):
        deltas.append(
            {
                "track_index": ref_st["track_index"],
                "ref_mean": ref_st["mean"],
                "mut_mean": mut_st["mean"],
                "delta_mean": round(mut_st["mean"] - ref_st["mean"], 4),
                "ref_max": ref_st["max"],
                "mut_max": mut_st["max"],
                "delta_max": round(mut_st["max"] - ref_st["max"], 4),
                "ref_peak_pos": ref_st["max_pos"],
                "mut_peak_pos": mut_st["max_pos"],
                "peak_shift": mut_st["max_pos"] - ref_st["max_pos"],
            }
        )
    return deltas


def _csv_from_rows(headers: List[str], rows: List[List[Any]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(headers)
    writer.writerows(rows)
    return buf.getvalue()


def _round_num(value: Any, digits: int = 6) -> Any:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


def _single_stats_csv(result: dict) -> str:
    """CSV of per-track segment stats for a single-sequence run."""
    rows: List[List[Any]] = []
    tissue = result.get("tissue_label") or ""
    ontology = result.get("tissue_ontology") or ""
    length = result.get("input_length")
    for st in result.get("segment_stats") or []:
        rows.append(
            [
                tissue,
                ontology,
                length,
                st.get("track_index"),
                _round_num(st.get("min")),
                _round_num(st.get("mean")),
                _round_num(st.get("max")),
                st.get("max_pos"),
            ]
        )
    return _csv_from_rows(
        ["tissue", "ontology", "length", "track", "min", "mean", "max", "peak_pos"],
        rows,
    )


def _compare_deltas_csv(compare_result: dict) -> str:
    """CSV of the Compare delta table (mutant − reference), including peak shift."""
    rows: List[List[Any]] = []
    tissue = compare_result.get("tissue_label") or ""
    ontology = compare_result.get("tissue_ontology") or ""
    length = compare_result.get("input_length")
    changed = ";".join(str(p) for p in (compare_result.get("diff_positions") or []))
    for d in compare_result.get("deltas") or []:
        rows.append(
            [
                tissue,
                ontology,
                length,
                changed,
                d.get("track_index"),
                _round_num(d.get("ref_mean")),
                _round_num(d.get("mut_mean")),
                _round_num(d.get("delta_mean"), 4),
                _round_num(d.get("ref_max")),
                _round_num(d.get("mut_max")),
                _round_num(d.get("delta_max"), 4),
                d.get("ref_peak_pos"),
                d.get("mut_peak_pos"),
                d.get("peak_shift"),
            ]
        )
    return _csv_from_rows(
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
        rows,
    )


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


@app.route("/", methods=["GET", "POST"])
def index():
    sequence: str = ""
    reference: str = ""
    mutant: str = ""
    mode: str = "single"
    tissue: str = DEFAULT_TISSUE
    result: Optional[dict] = None
    compare_result: Optional[dict] = None
    error: Optional[str] = None

    if request.method == "POST":
        mode = (request.form.get("mode") or "single").strip()
        sequence = (request.form.get("sequence") or "").strip().upper()
        reference = (request.form.get("reference") or "").strip().upper()
        mutant = (request.form.get("mutant") or "").strip().upper()
        tissue = (request.form.get("tissue") or DEFAULT_TISSUE).strip()
        if tissue not in TISSUE_OPTIONS:
            tissue = DEFAULT_TISSUE
        api_key = (request.form.get("api_key") or "").strip() or None

        if mode == "compare":
            err_ref = _validate_sequence(reference, "Reference")
            err_mut = _validate_sequence(mutant, "Mutant")
            if err_ref:
                error = err_ref
            elif err_mut:
                error = err_mut
            elif len(reference) != len(mutant):
                error = "Reference and mutant must be the same length."
            else:
                try:
                    model = get_model(api_key)
                    ref_data = _predict_dnase(model, reference, tissue)
                    mut_data = _predict_dnase(model, mutant, tissue)
                    deltas = _compare_stats(ref_data["segment_stats"], mut_data["segment_stats"])
                    compare_result = {
                        "ref": ref_data,
                        "mut": mut_data,
                        "deltas": deltas,
                        "diff_positions": _diff_positions(reference, mutant),
                        "input_length": len(reference),
                        "tissue_label": ref_data["tissue_label"],
                        "tissue_ontology": ref_data["tissue_ontology"],
                    }
                    compare_result["csv_text"] = _compare_deltas_csv(compare_result)
                except Exception as exc:  # noqa: BLE001
                    error = f"Error while calling AlphaGenome: {exc}"
        else:
            err = _validate_sequence(sequence)
            if err:
                error = err
            else:
                try:
                    model = get_model(api_key)
                    result = _predict_dnase(model, sequence, tissue)
                    result["csv_text"] = _single_stats_csv(result)
                except Exception as exc:  # noqa: BLE001
                    error = f"Error while calling AlphaGenome: {exc}"

    show_api_key_field = not bool(os.environ.get("ALPHA_GENOME_API_KEY"))
    return render_template(
        "index.html",
        sequence=sequence,
        reference=reference,
        mutant=mutant,
        mode=mode,
        tissue=tissue,
        tissue_options=TISSUE_OPTIONS,
        result=result,
        compare_result=compare_result,
        error=error,
        show_api_key_field=show_api_key_field,
    )


if __name__ == "__main__":
    # debug=True only for local experiments.
    app.run(host="127.0.0.1", port=5000, debug=True)

