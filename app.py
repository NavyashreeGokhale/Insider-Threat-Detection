"""
Insider Threat Detection Dashboard
Streamlit app for visualizing risk-scored data access events.

Run locally:  streamlit run app.py
Deploy:       push to GitHub, deploy via Streamlit Community Cloud
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from model import run_pipeline
from narratives import generate_incident_report, explain_event, recommend_action

st.set_page_config(page_title="Insider Threat Detection", layout="wide", page_icon="🛡️")

DATA_LOGS = "data/data_access_logs.csv"
DATA_PROFILES = "data/user_profiles.csv"


@st.cache_data
def load_scored_data():
    feats, _, _ = run_pipeline(DATA_LOGS, DATA_PROFILES)
    return feats


st.title("🛡️ Data Access Audit & Insider Threat Detection")
st.caption(
    "Behavioral anomaly detection on data access logs — Isolation Forest model "
    "+ rule-based risk boosts + explainable incident narratives."
)

feats = load_scored_data()

# ---------------- Top-line metrics ----------------
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Events", f"{len(feats):,}")
col2.metric("Flagged (Risk ≥ 50)", f"{(feats['risk_score'] >= 50).sum():,}",
            f"{(feats['risk_score'] >= 50).mean():.1%}")
col3.metric("CRITICAL", int((feats['severity'] == 'CRITICAL').sum()))
col4.metric("HIGH", int((feats['severity'] == 'HIGH').sum()))
col5.metric("Unique Users", feats['user_id'].nunique())

st.divider()

# ---------------- Sidebar filters ----------------
st.sidebar.header("Filters")
sev_filter = st.sidebar.multiselect(
    "Severity", options=['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'],
    default=['CRITICAL', 'HIGH']
)
dept_filter = st.sidebar.multiselect(
    "Department", options=sorted(feats['department'].dropna().unique()),
    default=[]
)
min_risk = st.sidebar.slider("Minimum risk score", 0, 100, 50)

filtered = feats[feats['risk_score'] >= min_risk]
if sev_filter:
    filtered = filtered[filtered['severity'].isin(sev_filter)]
if dept_filter:
    filtered = filtered[filtered['department'].isin(dept_filter)]

# ---------------- Charts ----------------
c1, c2 = st.columns(2)

with c1:
    sev_counts = feats['severity'].value_counts().reindex(['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']).fillna(0)
    fig = px.bar(
        x=sev_counts.index, y=sev_counts.values,
        color=sev_counts.index,
        color_discrete_map={'CRITICAL': '#b71c1c', 'HIGH': '#ef6c00', 'MEDIUM': '#fbc02d', 'LOW': '#66bb6a'},
        labels={'x': 'Severity', 'y': 'Event Count'},
        title="Alert Severity Distribution"
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    by_dept = feats.groupby('department')['risk_score'].mean().sort_values(ascending=False).reset_index()
    fig2 = px.bar(by_dept, x='department', y='risk_score',
                   title="Average Risk Score by Department",
                   labels={'risk_score': 'Avg Risk Score'})
    st.plotly_chart(fig2, use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    fig3 = px.histogram(feats, x='timestamp', y=None, nbins=50,
                          title="Access Events Over Time (all events)")
    st.plotly_chart(fig3, use_container_width=True)

with c4:
    risky = feats[feats['risk_score'] >= 50]
    fig4 = px.scatter(
        risky, x='timestamp', y='risk_score', color='severity',
        color_discrete_map={'CRITICAL': '#b71c1c', 'HIGH': '#ef6c00', 'MEDIUM': '#fbc02d', 'LOW': '#66bb6a'},
        hover_data=['username', 'department', 'action', 'resource'],
        title="Flagged Events Timeline"
    )
    st.plotly_chart(fig4, use_container_width=True)

st.divider()

# ---------------- Top alerts table ----------------
st.subheader(f"Top Alerts ({len(filtered)} matching filters)")

display_cols = ['timestamp', 'username', 'department', 'action', 'resource',
                 'resource_sensitivity', 'time_classification', 'risk_score', 'severity']
st.dataframe(
    filtered.sort_values('risk_score', ascending=False)[display_cols].reset_index(drop=True),
    use_container_width=True,
    height=350
)

st.divider()

# ---------------- Investigation panel ----------------
st.subheader("🔍 Investigation Toolkit")
st.caption("Select an event below to see its full explainable narrative and context.")

top_alerts = filtered.sort_values('risk_score', ascending=False).head(50)
if len(top_alerts) > 0:
    options = top_alerts.apply(
        lambda r: f"{r['timestamp']} | {r['username']} | {r['action']} on {r['resource']} | Risk {r['risk_score']:.0f}",
        axis=1
    ).tolist()
    selected = st.selectbox("Select an alert to investigate", options)
    sel_idx = options.index(selected)
    row = top_alerts.iloc[sel_idx]

    cc1, cc2 = st.columns([2, 1])
    with cc1:
        st.markdown(f"### Alert: {row['severity']} — Risk {row['risk_score']:.0f}/100")
        st.markdown(f"**User:** {row['username']} ({row['department']}, {row['job_title']}, {row['privilege_level']})")
        st.markdown(f"**Action:** `{row['action']}` on `{row['resource']}` ({row['resource_sensitivity']} sensitivity)")
        st.markdown(f"**Timestamp:** {row['timestamp']} ({row['time_classification'].replace('_', ' ')})")
        st.markdown(f"**Status:** {row['status']} | **Source IP:** {row['source_ip']}")

        st.markdown("**Anomaly Signals Detected:**")
        for reason in explain_event(row):
            st.markdown(f"- {reason}")

        st.info(f"**Recommended Action:** {recommend_action(row['severity'])}")

    with cc2:
        st.markdown("**User Baseline Context**")
        st.markdown(f"- Account inactive: {int(row['days_inactive'])} days")
        st.markdown(f"- % of access at night/unusual hours: {row['pct_night']:.1%}")
        st.markdown(f"- % weekend access: {row['pct_weekend']:.1%}")
        st.markdown(f"- % export actions: {row['pct_export']:.1%}")
        st.markdown(f"- Avg sensitivity accessed: {row['mean_sens']:.2f} / 3")
        st.markdown(f"- Total historical events: {int(row['total_events'])}")
else:
    st.info("No alerts match the current filters.")

st.divider()

# ---------------- Sample incident report ----------------
with st.expander("📄 Sample Incident Report (Top 15 Critical Alerts)"):
    report = generate_incident_report(feats, top_n=15)
    for _, r in report.iterrows():
        st.markdown(f"**[{r['severity']}] {r['username']} — Risk {r['risk_score']:.0f}/100**")
        st.write(r['narrative'])
        st.markdown("---")

st.caption(
    "Built for Problem Statement 04: Data Access Audit & Insider Threat Detection. "
    "Model: Isolation Forest (behavioral features) + rule-based risk boosts. "
    "Scaling notes available in TECHNICAL_DOCS.md"
)
