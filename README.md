# Phishing Detector

Ensemble ML classifier (XGBoost + Random Forest) predicting phishing website
risk from lexical URL features and host-based telemetry (DNS, WHOIS, SSL,
HTTP headers). See `Research Proposal.pdf` / `research outline.pdf` for the
full methodology and timeline.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project layout

```
data/
  raw/         downloaded feeds (gitignored, dated filenames)
  processed/   combined labeled dataset.csv (gitignored)
  cache/       local cache for WHOIS/DNS/SSL lookups (gitignored)
src/
  data_collection/
    fetch_feeds.py    download OpenPhish, PhishTank, Majestic Million
    build_dataset.py  combine feeds into data/processed/dataset.csv
  features/
    lexical.py         URL length, entropy, keyword vectors, etc. (done)
    host_telemetry.py  DNS/WHOIS/SSL/headers (Week 2 — stubs for now)
  models/        baseline + XGBoost/RF ensemble training, evaluation
  utils/         shared helpers (e.g. local caching)
notebooks/       exploratory analysis, SHAP plots
tests/
```

## Week 1 (current)

```bash
python -m src.data_collection.fetch_feeds     # downloads raw feeds to data/raw/
python -m src.data_collection.build_dataset   # writes data/processed/dataset.csv
python -m src.features.lexical                # sanity-check feature extraction on a sample URL
```

Note: the PhishTank CSV endpoint is sometimes rate-limited or requires a free
API key for reliable access — if `fetch_feeds.py` fails on it, register for a
key at phishtank.org and swap the URL in `FEEDS` for the authenticated
endpoint.
