# Alphagenomio

Live URL: `https://alphagenomio.onrender.com/`

Research UI for [AlphaGenome](https://www.alphagenomedocs.com/) DNase predictions. For non-commercial use; bring your own API key.

## How to use

1. Open `https://alphagenomio.onrender.com/`.
2. Paste your AlphaGenome API key if the form asks for it (not stored on the server).
3. Choose a tissue (Lung / Liver / Brain) if you want something other than lung.
4. Choose a mode:
   - **Single sequence** — paste DNA (A/C/G/T/N), or click **Insert example**, then **Run AlphaGenome**.
   - **Compare ref vs mutant** — paste reference and mutant (same length), or click **Insert example**, then **Compare**.
5. Review stats, peak highlights, and the delta table (mutant − reference).
6. Click **Download CSV** to save the delta table (Compare) or segment stats (single run). No second API call.

See [vision.md](vision.md) for project goals and next steps.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000/

Tests (mocked AlphaGenome, no network, no secrets):

```bash
python -m unittest test_app.py
```

## Redeploy on Render

Render is expected to auto-deploy from `main`. After this branch is merged:

1. Confirm the GitHub commit is on `main`.
2. Wait for the Render deploy of that commit (Dashboard → Alphagenomio service → Events / Deploys).
3. Hard-refresh `https://alphagenomio.onrender.com/` and check: tissue dropdown, Insert example, Compare, **Download CSV** after a run.

If auto-deploy is off, trigger **Manual Deploy → Deploy latest commit** on that service. Do not put API keys in the repo; keep `ALPHA_GENOME_API_KEY` (optional server-side key) and user keys in Render env / the form only.
