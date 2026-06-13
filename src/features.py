"""
Feature engineering for Insider Threat Detection (Problem Statement 04)
Builds per-event and per-user-baseline features used by the Isolation Forest model.
"""
import pandas as pd
import numpy as np

SENSITIVITY_MAP = {'low': 0, 'medium': 1, 'high': 2, 'restricted': 3}
TIME_CLASS_RISK = {'business_hours': 0, 'weekend': 1, 'unusual_hours': 2, 'night': 3}
ACTION_RISK = {'login': 0, 'file_access': 1, 'sql_query': 1, 'api_call': 1,
               'export_data': 3, 'admin_operation': 2}


def load_data(logs_path, profiles_path):
    logs = pd.read_csv(logs_path, parse_dates=['timestamp'])
    profiles = pd.read_csv(profiles_path, parse_dates=['last_login', 'hire_date'])
    return logs, profiles


def build_user_baselines(logs: pd.DataFrame) -> pd.DataFrame:
    """Per-user baseline statistics computed over their full history."""
    df = logs.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['dow'] = df['timestamp'].dt.dayofweek
    df['sens_score'] = df['resource_sensitivity'].map(SENSITIVITY_MAP)

    baselines = df.groupby('user_id').agg(
        total_events=('timestamp', 'count'),
        n_unique_resources=('resource', 'nunique'),
        mean_sens=('sens_score', 'mean'),
        max_sens=('sens_score', 'max'),
        pct_night=('time_classification', lambda x: (x.isin(['night', 'unusual_hours'])).mean()),
        pct_weekend=('time_classification', lambda x: (x == 'weekend').mean()),
        pct_export=('action', lambda x: (x == 'export_data').mean()),
        pct_failure=('status', lambda x: (x == 'failure').mean()),
        first_seen=('timestamp', 'min'),
        last_seen=('timestamp', 'max'),
    ).reset_index()

    # Set of resources each user has ever accessed (for "first-time access" detection)
    resource_sets = df.groupby('user_id')['resource'].apply(set).to_dict()
    baselines['_resource_set'] = baselines['user_id'].map(resource_sets)

    return baselines


def engineer_event_features(logs: pd.DataFrame, profiles: pd.DataFrame,
                             baselines: pd.DataFrame) -> pd.DataFrame:
    """
    For each access event, compute features describing how it deviates
    from that user's established baseline + role expectations.
    """
    df = logs.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['dow'] = df['timestamp'].dt.dayofweek
    df['sens_score'] = df['resource_sensitivity'].map(SENSITIVITY_MAP).fillna(0)
    df['time_risk'] = df['time_classification'].map(TIME_CLASS_RISK).fillna(0)
    df['action_risk'] = df['action'].map(ACTION_RISK).fillna(0)

    df = df.merge(profiles[['user_id', 'department', 'job_title', 'privilege_level',
                             'systems_access', 'days_inactive', 'is_active']],
                   on='user_id', how='left', suffixes=('', '_profile'))

    df = df.merge(baselines[['user_id', 'pct_night', 'pct_weekend', 'pct_export',
                              'pct_failure', 'mean_sens', 'max_sens', 'total_events',
                              '_resource_set']], on='user_id', how='left')

    # --- Engineered signal features ---

    # 1. Is this event night/unusual but user RARELY operates at night?
    df['is_offhours'] = df['time_classification'].isin(['night', 'unusual_hours']).astype(int)
    df['offhours_deviation'] = df['is_offhours'] * (1 - df['pct_night'].fillna(0))

    # 2. Is this an export, but user rarely exports?
    df['is_export'] = (df['action'] == 'export_data').astype(int)
    df['export_deviation'] = df['is_export'] * (1 - df['pct_export'].fillna(0))

    # 3. Sensitivity above user's typical access level
    df['sens_above_baseline'] = (df['sens_score'] - df['mean_sens'].fillna(0)).clip(lower=0)

    # 4. First-time access to this resource for this user
    def first_time(row):
        rs = row['_resource_set']
        if not isinstance(rs, set):
            return 0
        return 0 if row['resource'] in rs else 1
    # Note: _resource_set includes the current event; recompute leave-one-out approximation
    df['is_first_time_resource'] = 0  # placeholder, refined below

    # Proper leave-one-out: count resource occurrences per user up to (not incl.) this event
    df = df.sort_values('timestamp').reset_index(drop=True)
    seen = {}
    first_time_flags = []
    for _, row in df.iterrows():
        uid, res = row['user_id'], row['resource']
        s = seen.setdefault(uid, set())
        first_time_flags.append(1 if res not in s else 0)
        s.add(res)
    df['is_first_time_resource'] = first_time_flags

    # 5. Privilege mismatch: junior/user accessing restricted/high-sensitivity data
    df['priv_mismatch'] = ((df['privilege_level'].isin(['user'])) &
                            (df['sens_score'] >= 2)).astype(int)

    # 6. Failed login / failed access attempts (possible credential compromise signal)
    df['is_failure'] = (df['status'] == 'failure').astype(int)

    # 7. Cross-department access proxy: HRIS/GL_System/Customer_Vault accessed by
    #    users outside Finance/HR/Sales/Compliance respectively
    cross_dept_map = {
        'HRIS': ['HR', 'Executive'],
        'GL_System': ['Finance', 'Executive'],
        'Customer_Vault': ['Sales', 'Support', 'Marketing', 'Executive'],
        'PROD_DB': ['Engineering', 'IT', 'Operations'],
        'Admin_Console': ['IT', 'Security', 'Engineering'],
        'SIEM': ['Security', 'IT'],
    }
    def cross_dept(row):
        allowed = cross_dept_map.get(row['resource'])
        if allowed is None:
            return 0
        return 0 if row['department'] in allowed else 1
    df['cross_dept_access'] = df.apply(cross_dept, axis=1)

    # 8. Stale/inactive account activity
    df['stale_account_flag'] = (df['days_inactive'].fillna(0) > 30).astype(int)

    # 9. Admin operation by non-admin
    df['admin_op_by_nonadmin'] = ((df['action'] == 'admin_operation') &
                                   (~df['privilege_level'].isin(['admin', 'power-user']))).astype(int)

    # 10. Weekend + export combo
    df['weekend_export'] = ((df['time_classification'] == 'weekend') &
                             (df['is_export'] == 1)).astype(int)

    df = df.drop(columns=['_resource_set'], errors='ignore')
    return df


FEATURE_COLUMNS = [
    'sens_score', 'time_risk', 'action_risk',
    'offhours_deviation', 'export_deviation', 'sens_above_baseline',
    'is_first_time_resource', 'priv_mismatch', 'is_failure',
    'cross_dept_access', 'stale_account_flag', 'admin_op_by_nonadmin',
    'weekend_export',
]


if __name__ == '__main__':
    logs, profiles = load_data('../data/data_access_logs.csv', '../data/user_profiles.csv')
    baselines = build_user_baselines(logs)
    feats = engineer_event_features(logs, profiles, baselines)
    print(feats.shape)
    print(feats[FEATURE_COLUMNS].describe())
    feats.to_csv('../output/engineered_features.csv', index=False)
    print("Saved engineered_features.csv")
