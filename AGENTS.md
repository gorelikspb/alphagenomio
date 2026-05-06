# Agents

## Cursor Cloud specific instructions

### Project overview

Alphagenomio is a single-service Python/Flask web app that wraps the Google DeepMind AlphaGenome API. There is no database, no Docker, no build step, and no test suite.

### Running the dev server

```bash
python app.py
```

Starts Flask on `http://127.0.0.1:5000` with `debug=True` (hot-reload enabled).

### External dependency

The app requires a valid `ALPHA_GENOME_API_KEY` to make real API calls. Without it, the app still starts and serves the form, but submissions return an error. The key can be set as an environment variable or entered per-request in the web UI.

### Gotchas

- `pip install -r requirements.txt` may fail on the system `packaging` package. Use `pip install --break-system-packages --ignore-installed packaging -r requirements.txt` if that happens.
- There are no automated tests, no linter configuration, and no type-checking config in this repo.
- Dependencies in `requirements.txt` are unpinned.
- When `ALPHA_GENOME_API_KEY` is set as an env var, the web UI hides the API key input field and uses the server-side key for all requests. When unset, the field appears and users supply it per-request.
- AlphaGenome API keys can expire. If you see a gRPC `INVALID_ARGUMENT` / "API key expired" error, the key needs renewal at the Google DeepMind console.
