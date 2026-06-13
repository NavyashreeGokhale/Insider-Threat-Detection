"""
Anomaly detection model + risk scoring for Insider Threat Detection.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLUMNS, load_data, build_user_baselines, engineer_event_features


def train_isolation_forest(feats: pd.DataFrame, contamination=0.10, random_state=42):
    X = feats[FEATURE_COLUMNS].fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(Xs)

    # decision_function: higher = more normal, lower/negative = more anomalous
    raw_scores = model.decision_function(Xs)
    return model, scaler, raw_scores


def normalize_to_risk_score(raw_scores: np.ndarray) -> np.ndarray:
    """
    Convert isolation forest decision_function scores to a 0-100 risk scale
    where higher = riskier. Uses percentile-rank normalization so that the
    bulk of "normal" events cluster near the low end, and only genuinely
    anomalous tails push toward 100 (avoids the median sitting at ~50).
    """
    inverted = -raw_scores  # flip so higher = more anomalous
    ranks = pd.Series(inverted).rank(pct=True).values  # 0..1, uniform
    # Power curve compresses the bottom ~70% of events (normal) toward
    # low scores, while the anomalous tail stretches toward 100.
    scaled = np.power(ranks, 2.5) * 100
    return scaled


def apply_rule_boosts(feats: pd.DataFrame, base_risk: np.ndarray) -> np.ndarray:
    """
    Combine ML-based risk with explicit rule-based boosts for known
    high-severity patterns. Caps at 100.
    """
    risk = base_risk.copy()

    # Critical combo: off-hours + export + restricted/high sensitivity
    critical_combo = (
        (feats['is_offhours'] == 1) &
        (feats['is_export'] == 1) &
        (feats['sens_score'] >= 2)
    )
    risk[critical_combo] += 20

    # First-time access to high/restricted resource
    first_time_sensitive = (feats['is_first_time_resource'] == 1) & (feats['sens_score'] >= 2)
    risk[first_time_sensitive] += 8

    # Stale account performing any export/admin op
    stale_active = (feats['stale_account_flag'] == 1) & (feats['action'].isin(['export_data', 'admin_operation']))
    risk[stale_active] += 12

    # Cross-department access to restricted data
    cross_dept_sensitive = (feats['cross_dept_access'] == 1) & (feats['sens_score'] >= 2)
    risk[cross_dept_sensitive] += 10

    # Privilege mismatch
    risk[feats['priv_mismatch'] == 1] += 8

    # Admin op by non-admin
    risk[feats['admin_op_by_nonadmin'] == 1] += 6

    # Failed access attempts (possible credential probing)
    risk[feats['is_failure'] == 1] += 6

    return np.clip(risk, 0, 100)


def severity_band(score):
    if score >= 80:
        return 'CRITICAL'
    elif score >= 60:
        return 'HIGH'
    elif score >= 35:
        return 'MEDIUM'
    else:
        return 'LOW'


def run_pipeline(logs_path, profiles_path):
    logs, profiles = load_data(logs_path, profiles_path)
    baselines = build_user_baselines(logs)
    feats = engineer_event_features(logs, profiles, baselines)

    model, scaler, raw_scores = train_isolation_forest(feats)
    base_risk = normalize_to_risk_score(raw_scores)
    final_risk = apply_rule_boosts(feats, base_risk)

    feats['ml_risk_base'] = base_risk
    feats['risk_score'] = final_risk.round(1)
    feats['severity'] = feats['risk_score'].apply(severity_band)
    feats['is_anomaly_pred'] = (feats['risk_score'] >= 50).astype(int)

    return feats, model, scaler


if __name__ == '__main__':
    feats, model, scaler = run_pipeline('../data/data_access_logs.csv', '../data/user_profiles.csv')

    print("Severity distribution:")
    print(feats['severity'].value_counts())
    print()
    print("Anomaly flag rate:", feats['is_anomaly_pred'].mean())
    print()
    print("Top 10 highest risk events:")
    top = feats.sort_values('risk_score', ascending=False).head(10)
    print(top[['timestamp', 'username', 'department', 'action', 'resource',
                'resource_sensitivity', 'time_classification', 'risk_score', 'severity']].to_string(index=False))

    feats.to_csv('../output/scored_events.csv', index=False)
    print("\nSaved scored_events.csv")
