# Technical Documentation — Insider Threat Detection System

## 1. Architecture Overview

```
data_access_logs.csv ─┐
                       ├─> features.py  ──> engineered feature matrix (13 behavioral features)
user_profiles.csv ────┘
                              │
                              ▼
                        model.py
              ┌──────────────────────────────┐
              │ Isolation Forest (unsupervised)│  ── ml_risk_base (0-100, percentile-ranked)
              │ + Rule-based risk boosts       │  ── critical combos, privilege mismatch, etc.
              └──────────────────────────────┘
                              │
                              ▼
                   final risk_score (0-100) + severity band
                              │
                              ▼
                       narratives.py
              (template-based explainable narratives,
               LLM-prompt-ready for future upgrade)
                              │
                              ▼
                         app.py (Streamlit dashboard)
```

## 2. Feature Engineering

For every access event, we compute features that describe **deviation from
that user's own established baseline**, not just absolute thresholds —
this is what allows the system to distinguish "DB admin working at night
(normal for them)" from "Marketing intern exporting Customer_Vault at 3 AM
(abnormal for them)".

| Feature | Description |
|---|---|
| `sens_score` | Numeric encoding of resource sensitivity (low=0 … restricted=3) |
| `time_risk` | Numeric encoding of time classification (business_hours=0 … night=3) |
| `action_risk` | Risk weight of the action type (export=3, admin_op=2, etc.) |
| `offhours_deviation` | Off-hours access × (1 − user's historical % off-hours access) |
| `export_deviation` | Export action × (1 − user's historical % exports) |
| `sens_above_baseline` | How much more sensitive this access is vs. the user's average |
| `is_first_time_resource` | 1 if user has never accessed this resource before (leave-one-out, time-ordered) |
| `priv_mismatch` | User-tier privilege accessing high/restricted data |
| `is_failure` | Failed access attempt (credential probing signal) |
| `cross_dept_access` | Access to a resource not typically used by the user's department |
| `stale_account_flag` | Account inactive >30 days, now active |
| `admin_op_by_nonadmin` | Admin operation performed by non-admin/power-user |
| `weekend_export` | Export action occurring on a weekend |

## 3. Model: Isolation Forest + Rule-Based Risk Boosts

**Why hybrid (Option A spirit)?** Pure ML (Isolation Forest) finds
*statistical* outliers but doesn't know that "export + restricted data +
night + first-time access" is categorically more dangerous than four
independent minor deviations. Pure rules miss novel combinations.
We combine both:

1. **Isolation Forest** (200 trees, contamination=0.10) trained on the
   13-feature matrix (standardized). Produces a `decision_function` score.
2. **Percentile-rank normalization**: raw IF scores are converted to
   percentile ranks, then passed through a power curve (`rank^2.5 * 100`).
   This ensures *normal* behavior clusters near 0-30, while only the
   genuinely anomalous tail approaches 100 — avoiding the common pitfall
   where min-max scaling pushes the median to ~50.
3. **Rule-based boosts** (additive, capped at 100):
   - Off-hours + export + high/restricted sensitivity: **+20**
   - First-time access to high/restricted resource: **+8**
   - Stale account performing export/admin op: **+12**
   - Cross-department access to restricted data: **+10**
   - Privilege mismatch (user-tier on high-sensitivity): **+8**
   - Admin operation by non-admin: **+6**
   - Failed access attempt: **+6**

**Severity bands:**
- CRITICAL: ≥80
- HIGH: 60-79
- MEDIUM: 35-59
- LOW: <35

**Anomaly flag threshold:** risk_score ≥ 50 → flagged for review.

## 4. Results on Sample Data (1,200 events, 100 users, 365 days)

| Severity | Count | % |
|---|---|---|
| CRITICAL | 212 | 17.7% |
| HIGH | 117 | 9.8% |
| MEDIUM | 194 | 16.2% |
| LOW | 677 | 56.4% |
| **Flagged (≥50)** | **382** | **31.8%** |

**Feature signal strength** (avg risk score with vs. without signal):
- `is_failure`: 76.1 vs 35.6 — strongest single discriminator
- `admin_op_by_nonadmin`: 75.1 vs 32.5
- `priv_mismatch`: 64.6 vs 29.4
- `is_export`: 65.2 vs 32.7
- `is_offhours`: 60.8 vs 29.9

This confirms the engineered features are doing real discriminative work,
not just adding noise.

## 5. False Positive Control (Edge Cases)

The problem statement highlights several legitimate-but-unusual patterns.
Here's how the design accounts for them:

| Edge Case | How It's Handled |
|---|---|
| **Seasonal bulk access (month-end)** | `offhours_deviation` and `export_deviation` are *relative* to the user's own history, so a Finance analyst who regularly does month-end exports has a high `pct_export` baseline → low deviation score, even if absolute export volume is high. |
| **Role changes / new admin** | `priv_mismatch` and `cross_dept_access` only fire when current access contradicts the *current* profile (privilege_level, department) — a newly promoted admin won't trigger `admin_op_by_nonadmin`. |
| **On-call rotation (elevated access)** | Because deviation is baseline-relative and the IF model learns from the full 365-day history, repeated on-call patterns become part of the "normal" distribution for that user over time. |
| **Contractors (short tenure)** | `stale_account_flag` and first-time-resource flags will naturally fire more for new accounts — this is *intentional*, since short-tenure + sensitive access is a real elevated-risk pattern worth surfacing, even if often legitimate. We recommend a lower severity weight for contractor accounts in production (configurable). |
| **Service accounts (no "normal" pattern)** | Service accounts (`privilege_level == 'service-account'`) tend to have stable, repetitive patterns; the IF model handles this naturally since their feature vectors cluster tightly — low variance = low anomaly score for routine service-account behavior. |

**Current limitation:** with only ~31.8% of events flagged at risk≥50 on
this synthetic dataset (which has high inherent anomaly density by design),
absolute precision/recall against a ground-truth label set could not be
computed (no `data_access_labels.csv` / `user_profile_labels.csv` was
provided with this dataset — only the raw access logs and user profiles).
The system instead exposes a **tunable risk threshold** (sidebar slider)
so analysts can calibrate the flag rate to their tolerance, and every flag
includes a full **explainability narrative** for rapid triage.

## 6. Scaling to 1M+ Daily Events

The current implementation runs the full pipeline in-memory on 1,200
events in under a second. For production scale:

**Ingestion & Streaming**
- Replace CSV batch load with a streaming ingestion layer (Kafka /
  Kinesis) — each access event becomes a message on an `access-events` topic.
- Use a stream processor (Spark Structured Streaming / Flink) to compute
  rolling per-user feature aggregates (e.g. windowed `pct_night`,
  `pct_export`) incrementally rather than recomputing over full history.

**Storage & Partitioning**
- Partition the access log store (e.g. Delta Lake / Iceberg on S3) by
  `date` and `department` for efficient time-range and department-scoped
  queries.
- Maintain a separate, small **user baseline table** (one row per user,
  updated incrementally) — this is the table the real-time scorer joins
  against, avoiding full-history scans per event.

**Model Serving**
- Train the Isolation Forest offline/periodically (e.g. nightly batch job
  on the previous day's full event volume) and serialize with `joblib`.
- Serve scoring via a lightweight microservice (FastAPI) or as a Spark UDF
  for batch scoring — each event scores in microseconds given precomputed
  baselines.
- Rule-based boosts remain stateless and trivially parallelizable (pure
  function of event + baseline row).

**Distributed Compute Estimate**
- At 1M events/day (~11.6 events/sec average, with peaks), feature
  computation + scoring per event is O(1) given precomputed baselines —
  a single FastAPI instance can handle this; horizontal scaling via
  multiple replicas behind a load balancer handles peak bursts.
- Baseline table updates (daily aggregation) on 1M events/day is a
  standard Spark batch job, completing well within minutes on a small
  cluster (4-8 executors).

**Alerting & Dashboard**
- Push CRITICAL/HIGH alerts to a message queue → SIEM integration (Splunk,
  Sentinel) for real-time analyst notification (<5 min target).
- Dashboard (Streamlit prototype here) would be replaced/backed by a
  proper OLAP store (ClickHouse / Druid) for sub-second aggregation
  queries over millions of rows.

## 7. DLP Integration (Bonus)

For real exfiltration prevention, CRITICAL-severity events involving
`export_data` to `destination` values like `usb_drive` or
`external_email` (where available in richer log schemas) would trigger a
synchronous webhook to a DLP policy engine *before* the export completes
— blocking it pending analyst review, rather than alerting after the fact.
This requires the access control layer to call the scoring service
inline (pre-action), which is feasible since per-event scoring is O(1)
given precomputed baselines.
