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
    fetch_feeds.py         download OpenPhish, PhishTank, Majestic Million
    build_dataset.py       combine feeds into data/processed/dataset.csv
    build_host_features.py run host telemetry over the dataset -> host_features.csv
  features/
    lexical.py         URL length, entropy, keyword vectors, etc.
    host_telemetry.py  DNS/WHOIS/SSL/HTTP header lookups, cached per-host
  models/        baseline + XGBoost/RF ensemble training, evaluation
  utils/         shared helpers (e.g. local caching)
notebooks/       exploratory analysis, SHAP plots
tests/
```

## Week 1 — dataset + lexical features

```bash
python -m src.data_collection.fetch_feeds     # downloads raw feeds to data/raw/
python -m src.data_collection.build_dataset   # writes data/processed/dataset.csv
python -m src.features.lexical                # sanity-check feature extraction on a sample URL
```

Note: the PhishTank CSV endpoint is sometimes rate-limited or requires a free
API key for reliable access — if `fetch_feeds.py` fails on it, register for a
key at phishtank.org and swap the URL in `FEEDS` for the authenticated
endpoint.

## Week 2 — host-based telemetry

```bash
python -m src.data_collection.build_host_features 200   # sample of 200 URLs
python -m src.data_collection.build_host_features 0     # 0 = full dataset (slow, see below)
```

Extracts DNS record flags (A/MX/TXT), WHOIS domain age, SSL certificate
validity, and HTTP response headers for each URL, and merges them with the
lexical features into `data/processed/host_features.csv`.

Every lookup is cached to disk under `data/cache/` (keyed by hostname), so
re-running the command only re-queries hosts not seen in the last 7 days.
Concurrency is capped at 50 in-flight requests (`CONCURRENCY` in
`build_host_features.py`) to stay under WHOIS/DNS rate limits — running the
full ~126k-row dataset will still take a long time and risks WHOIS IP
throttling, so start with a sample (a few hundred to a few thousand rows) for
model development and only scale up once the pipeline is stable.
