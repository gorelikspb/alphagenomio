# Alphagenomio

Live URL: `https://alphagenomio.onrender.com/`

Research UI for [AlphaGenome](https://www.alphagenomedocs.com/) DNase predictions. For non-commercial use; bring your own API key.

## How to use

1. Open `https://alphagenomio.onrender.com/`.
2. Paste your AlphaGenome API key if the form asks for it (not stored on the server).
3. Choose a mode:
   - **Single sequence** — paste DNA (A/C/G/T/N) and click **Run AlphaGenome**.
   - **Compare ref vs mutant** — paste reference and mutant (same length), click **Compare**.
4. Review stats, peak highlights, and the delta table (mutant − reference).

See [vision.md](vision.md) for project goals and next steps.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000/
