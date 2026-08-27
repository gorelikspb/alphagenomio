# Alphagenomio

Live URL: `https://alphagenomio.onrender.com/`

Independent research prototype for trying [AlphaGenome](https://www.alphagenomedocs.com/) DNase predictions. Not a Google or DeepMind product. The AlphaGenome API is for non-commercial research use only.

## How to use

1. Open the live URL, or run locally (below).
2. Paste your AlphaGenome API key if the form asks for it (BYOK). The key is sent only with that request and is not stored. If `ALPHA_GENOME_API_KEY` is set on the server, the field stays hidden.
3. Choose a tissue (default: lung, `UBERON:0002048`).
4. Paste a **Reference** DNA sequence (A/C/G/T/N).
5. Click **Run AlphaGenome** for a single-sequence DNase summary, or paste a **Mutant** sequence and click **Compare** for track-0 Δmean, Δmax, and peak shift.

Both sequences in a compare use the same tissue and DNase output settings.

## Local run

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000/`.

To exercise the form without a real API key (local UI only):

```bash
ALPHAGENOME_MOCK=1 python app.py
```

Do not enable `ALPHAGENOME_MOCK` on the public Render service.

Tests (no AlphaGenome network calls):

```bash
python -m unittest test_app.py
```
