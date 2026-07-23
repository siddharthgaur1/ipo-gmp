# Security

## Threat model

IPO GMP Predictor is a self-contained ML dashboard. It has no authentication, no
user-uploaded files, no LLM, no secrets, and makes no network calls at runtime.
The dataset is **synthetic** (deterministically generated) and the models are
trained from it on first boot. The attack surface is the Streamlit process.

## What is mitigated

| Risk | Status | Notes |
|---|---|---|
| Secrets in git history | **Clean** — `gitleaks`: 0 findings; no `.env`, no keys anywhere |
| Dependency CVEs | **Clean** — `pip-audit`: no known vulnerabilities; versions pinned |
| Untrusted `pickle.load` | **Mitigated by construction** — the only `.pkl` files loaded are the regressor/classifier this app **trains itself on boot** (`_bootstrap()` in `src/app.py`). They are gitignored, never downloaded, and never user-supplied. The app never unpickles an artifact it did not just produce. |
| Code execution / SQL injection | **Not applicable** — no `eval`/`exec`/`subprocess`; SQLite reads are the app's own generated data with no user-controlled SQL |
| Data provenance | **Synthetic, and labelled as such** — `scripts/seed_data.py` generates the dataset (RNG seeded to 42), modelled after real-world distributions but not real IPO records |

## What is NOT mitigated / notes

- **No authentication.** Public read-only dashboard.
- **The metrics are measured on synthetic data.** MAE ≈ 12.7%, R² ≈ 0.87,
  directional accuracy ≈ 91% are the model's real cross-validation results — but on
  a synthetic dataset, so they demonstrate the *pipeline* (feature engineering,
  time-series CV, calibrated confidence bands), not real-market predictive power.
  This is stated in the README and the in-app Model Info tab.
- **`pickle` is safe here only because provenance is controlled.** If you ever
  point `MODEL_DIR` at models from an untrusted source, `pickle.load` becomes an
  arbitrary-code-execution vector — do not.

## Reporting

Open an issue. Portfolio/demo project, no production deployment, no security SLA.
