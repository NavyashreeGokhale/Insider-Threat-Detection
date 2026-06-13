"""
Narrative generation for flagged access events.

Produces human-readable, investigator-friendly explanations of WHY an
event was flagged, in the style of an LLM-generated incident narrative.
Implemented as a deterministic template engine (no external API needed
for the hackathon demo), but structured so the same feature payload can
be dropped into a real LLM prompt later (see `build_llm_prompt`).
"""
import pandas as pd


def explain_event(row: pd.Series) -> list:
    """Return a list of human-readable reasons this event was flagged."""
    reasons = []

    if row['is_offhours'] == 1:
        time_label = row['time_classification'].replace('_', ' ')
        if row['offhours_deviation'] > 0.5:
            reasons.append(
                f"Access during {time_label} — "
                f"unusual for {row['username']}, who rarely operates outside business hours"
            )
        else:
            reasons.append(f"Access occurred during {time_label}")

    if row['is_export'] == 1:
        if row['export_deviation'] > 0.5:
            reasons.append(
                f"Data export action — a rare event type for this user "
                f"(export deviation score: {row['export_deviation']:.2f})"
            )
        else:
            reasons.append("Data export action")

    if row['resource_sensitivity'] in ('high', 'restricted'):
        reasons.append(f"Target resource '{row['resource']}' is classified as {row['resource_sensitivity']}-sensitivity")

    if row['is_first_time_resource'] == 1:
        reasons.append(f"First-time access to '{row['resource']}' by this user")

    if row['priv_mismatch'] == 1:
        reasons.append(
            f"Privilege mismatch: user holds '{row['privilege_level']}' access tier "
            f"but accessed {row['resource_sensitivity']}-sensitivity data"
        )

    if row['cross_dept_access'] == 1:
        reasons.append(
            f"Cross-department access: {row['department']} user accessed "
            f"'{row['resource']}', typically restricted to other departments"
        )

    if row['stale_account_flag'] == 1:
        reasons.append(
            f"Account inactive for {int(row['days_inactive'])} days prior to this access "
            f"(dormant account reactivation risk)"
        )

    if row['admin_op_by_nonadmin'] == 1:
        reasons.append(f"Administrative operation performed by a non-admin user ({row['privilege_level']})")

    if row['is_failure'] == 1:
        reasons.append("Access attempt resulted in failure (possible credential probing)")

    if row['weekend_export'] == 1:
        reasons.append("Bulk data export occurred on a weekend")

    if not reasons:
        reasons.append("Statistical deviation from user's established access pattern (model-detected)")

    return reasons


def recommend_action(severity: str) -> str:
    return {
        'CRITICAL': 'BLOCK + IMMEDIATE INVESTIGATION + escalate to security team',
        'HIGH': 'INVESTIGATE within 24 hours + verify business justification',
        'MEDIUM': 'REVIEW in next analyst triage cycle',
        'LOW': 'MONITOR (no immediate action)',
    }.get(severity, 'MONITOR')


def build_narrative(row: pd.Series) -> str:
    reasons = explain_event(row)
    reason_text = "; ".join(reasons)
    return (
        f"{row['username']} ({row['department']}, {row['privilege_level']}) performed "
        f"a '{row['action']}' on '{row['resource']}' at {row['timestamp']}. "
        f"Risk score {row['risk_score']:.0f}/100 ({row['severity']}). "
        f"Key factors: {reason_text}. "
        f"Recommended action: {recommend_action(row['severity'])}."
    )


def build_llm_prompt(row: pd.Series) -> str:
    """
    Structured prompt for a real LLM (e.g. Claude/GPT) to generate a richer
    narrative. Kept here so the system can be upgraded from templates to a
    live LLM call by swapping `build_narrative` for an API call using this prompt.
    """
    reasons = explain_event(row)
    return (
        "You are a security analyst assistant. Given the following data access "
        "event and the anomaly signals detected, write a 2-3 sentence incident "
        "narrative explaining the risk in plain language for a non-technical "
        "investigator, and end with a recommended action.\n\n"
        f"User: {row['username']} | Department: {row['department']} | "
        f"Privilege: {row['privilege_level']}\n"
        f"Action: {row['action']} on {row['resource']} "
        f"({row['resource_sensitivity']} sensitivity)\n"
        f"Timestamp: {row['timestamp']} | Time class: {row['time_classification']}\n"
        f"Risk score: {row['risk_score']:.0f}/100 | Severity: {row['severity']}\n"
        f"Detected anomaly signals: {'; '.join(reasons)}\n"
    )


def generate_incident_report(feats: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    top = feats.sort_values('risk_score', ascending=False).head(top_n).copy()
    top['narrative'] = top.apply(build_narrative, axis=1)
    top['recommendation'] = top['severity'].apply(recommend_action)
    return top


if __name__ == '__main__':
    from model import run_pipeline
    feats, _, _ = run_pipeline('../data/data_access_logs.csv', '../data/user_profiles.csv')
    report = generate_incident_report(feats, top_n=15)
    for _, r in report.iterrows():
        print(r['narrative'])
        print('-' * 80)
    report.to_csv('../output/incident_report.csv', index=False)
