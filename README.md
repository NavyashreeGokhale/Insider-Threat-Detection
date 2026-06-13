# 🛡️ Data Access Audit & Insider Threat Detection

Hackathon project for **Problem Statement 04** — detecting abnormal data
access patterns to flag insider threats before sensitive data leaves the
organization.

## Approach: Option A — Behavioral ML + Explainable Narratives

- **Feature engineering**: 13 behavioral features per access event, computed
  relative to each user's own historical baseline (not just absolute
  thresholds).
- **Model**: Isolation Forest (unsupervised anomaly detection) combined with
  rule-based risk boosts for known high-severity patterns (off-hours bulk
  export, privilege mismatch, cross-department access, etc.)
- **Risk scoring**: 0-100 scale, percentile-normalized, with severity bands
  (CRITICAL/HIGH/MEDIUM/LOW).
- **Explainability**: Every flagged event gets a plain-language narrative
  listing the specific anomaly signals detected — built as a template
  engine designed to be drop-in replaceable with a live LLM call
  (prompt template included in `src/narratives.py`).
- **Dashboard**: Interactive Streamlit app with filters, charts, and an
  investigation toolkit for analyst triage.

## Project Structure

```
.
├── app.py                  # Streamlit dashboard (main entrypoint)
├── requirements.txt
├── TECHNICAL_DOCS.md        # Architecture, scaling, FP analysis
├── data/
│   ├── data_access_logs.csv
│   └── user_profiles.csv
├── src/
│   ├── features.py          # Feature engineering
│   ├── model.py              # Isolation Forest + risk scoring
│   └── narratives.py         # Explainable narrative generation
└── output/
    ├── scored_events.csv
    └── incident_report.csv
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Run the Pipeline Standalone

```bash
cd src
python features.py    # generates output/engineered_features.csv
python model.py        # generates output/scored_events.csv
python narratives.py   # generates output/incident_report.csv
```

## Results Summary (1,200 events / 100 users / 365 days)

| Severity | Count | % |
|---|---|---|
| CRITICAL | 212 | 17.7% |
| HIGH | 117 | 9.8% |
| MEDIUM | 194 | 16.2% |
| LOW | 677 | 56.4% |

See `TECHNICAL_DOCS.md` for full architecture, false-positive handling
across edge cases (seasonality, role changes, on-call, contractors), and
how this scales to 1M+ daily events.
