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


def create_model(api_key: Optional[str] = None):
    """Create AlphaGenome DNA client. Uses api_key if given, else ALPHA_GENOME_API_KEY env."""
    if dna_client is None:
        raise RuntimeError(
            "alphagenome is not installed. Run `pip install -r requirements.txt` first."
        )
    key = api_key or os.environ.get("ALPHA_GENOME_API_KEY")
    if not key:
        raise RuntimeError(
            "No API key: enter it in the field below, or set ALPHA_GENOME_API_KEY on the server."
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
    result: Optional[dict] = None
    error: Optional[str] = None

    if request.method == "POST":
        sequence = (request.form.get("sequence") or "").strip().upper()
        api_key = (request.form.get("api_key") or "").strip() or None
        if not sequence:
            error = "Please paste a DNA sequence."
        elif any(ch not in "ACGTN" for ch in sequence):
            error = "Sequence should contain only A, C, G, T (and optionally N)."
        else:
            try:
                model = get_model(api_key)

                # Pad sequence to a valid length for AlphaGenome (1MB) using 'N'.
                padded = sequence.center(
                    dna_client.SEQUENCE_LENGTH_1MB, "N"  # type: ignore[attr-defined]
                )

                output = model.predict_sequence(
                    sequence=padded,
                    requested_outputs=[dna_client.OutputType.DNASE],  # type: ignore[attr-defined]
                    ontology_terms=["UBERON:0002048"],  # Lung, as in quick-start example.
                )

                dnase_values = output.dnase.values  # shape: (padded_length, num_tracks)
                # Average only over the user's sequence (center segment), not over N-padding
                start_idx = (len(padded) - len(sequence)) // 2
                segment = dnase_values[start_idx : start_idx + len(sequence)]
                means = summarize_dnase_predictions(segment)
                api_raw = _build_api_raw_output(output, segment, start_idx, len(sequence))
                try:
                    api_raw_json = json.dumps(api_raw, indent=2, ensure_ascii=False)
                except (TypeError, ValueError):
                    api_raw_json = '{"note": "Full API response could not be serialized to JSON."}'

                result = {
                    "input_length": len(sequence),
                    "padded_length": len(padded),
                    "num_tracks": len(means),
                    "track_means": [round(m, 4) for m in means[:10]],  # show first 10
                    "api_raw_json": api_raw_json,
                }
            except Exception as exc:  # noqa: BLE001
                error = f"Error while calling AlphaGenome: {exc}"

    show_api_key_field = not bool(os.environ.get("ALPHA_GENOME_API_KEY"))
    return render_template(
        "index.html",
        sequence=sequence,
        result=result,
        error=error,
        show_api_key_field=show_api_key_field,
    )


if __name__ == "__main__":
    # Debug=True только для локальных экспериментов.
    app.run(host="127.0.0.1", port=5000, debug=True)

