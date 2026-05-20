import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file

from segment_labels import SEGMENT_PROFILES, get_segment_profile

app = Flask(__name__)

# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------

MODEL_DIR = Path('models')
MODEL_LOADED = False
kmeans_model = None
scaler = None
pca_model = None
feature_names = None
cluster_profiles = {}

try:
    kmeans_model = pickle.load(open(MODEL_DIR / 'kmeans_model.pkl', 'rb'))
    scaler = pickle.load(open(MODEL_DIR / 'scaler.pkl', 'rb'))
    pca_model = pickle.load(open(MODEL_DIR / 'pca_model.pkl', 'rb'))
    feature_names = pickle.load(open(MODEL_DIR / 'feature_names.pkl', 'rb'))
    with open(MODEL_DIR / 'cluster_profiles.json') as f:
        cluster_profiles = json.load(f)
    MODEL_LOADED = True
    print("All models loaded successfully.")
except FileNotFoundError as e:
    print(f"WARNING: Model file missing — {e}")
    print("Run train.py first to generate model files.")


def build_retention_strategy(cluster_id: int, data: dict) -> str:
    tenure = float(data.get('tenure', 0))
    charges = float(data.get('MonthlyCharges', 0))
    contract = data.get('Contract', 'Month-to-month')
    internet = data.get('InternetService', 'Fiber optic')
    tech_support = data.get('TechSupport') == 'Yes'
    online_security = data.get('OnlineSecurity') == 'Yes'

    if cluster_id == 0:  # At-Risk New Adopters
        if contract != 'Month-to-month':
            return (f"Already on a {contract} contract — churn risk is lower than typical for "
                    f"this segment. Focus on service engagement: enrol in TechSupport or "
                    f"OnlineSecurity to deepen the relationship before renewal.")
        if tenure < 6:
            return (f"Critical window — at {int(tenure)} months this customer has not yet formed "
                    f"loyalty. Trigger a proactive call this week and offer a price-locked "
                    f"one-year contract with one add-on included before the cancellation "
                    f"decision is made.")
        if tech_support or online_security:
            return ("Already using add-on services — leverage that engagement. Offer a one-year "
                    "price lock at their current rate to convert service satisfaction into a "
                    "contract commitment before the churn window opens.")
        if charges > 90:
            return (f"Paying ${charges:.0f}/mo with no commitment — a price-lock guarantee "
                    f"(no increases for 12 months) is more compelling than a bundle upgrade "
                    f"at this spend level. Lead with price certainty, not added services.")
        return (f"Month-to-month on ${charges:.0f}/mo. Offer a price-locked annual contract "
                f"bundled with TechSupport or OnlineSecurity. Proactive outreach at 30 days "
                f"reduces early churn — the goal is to build perceived value before the "
                f"cancellation window opens.")

    if cluster_id == 1:  # Stable Budget Loyalists
        if internet == 'No':
            return ("No internet service — the primary opportunity here is upsell, not retention. "
                    "A bundled internet introduction offer at a promotional rate tied to a "
                    "contract extension could significantly increase this customer's lifetime value "
                    "with minimal churn risk.")
        if charges < 30:
            return (f"Very low spend (${charges:.0f}/mo) with a strong commitment profile. "
                    f"Retention investment is unnecessary — this customer is not at risk. "
                    f"The opportunity is upsell: a service tier or add-on offer increases "
                    f"lifetime value without disrupting their loyalty.")
        return ("Stable, committed customer with low churn risk — retention spend here has low "
                "ROI. Focus on upsell: a service upgrade or add-on offer is the highest-value "
                "action for this segment.")

    if cluster_id == 2:  # Uncommitted New Subscribers
        if contract != 'Month-to-month':
            return (f"Already on a {contract} contract — less urgent than typical for this "
                    f"segment. Focus on renewal timing: reach out 60 days before contract end "
                    f"with a loyalty offer to prevent passive churn at renewal.")
        if tenure < 6:
            return (f"Very early stage ({int(tenure)} months) — most responsive to new-customer "
                    f"loyalty offers. A first-year price lock or a free month with contract "
                    f"sign-up is the highest-ROI intervention at this tenure.")
        if tenure >= 12:
            return (f"At {int(tenure)} months on month-to-month, this customer has shown some "
                    f"loyalty but no commitment. Past the typical early-churn window — a "
                    f"targeted upgrade offer with a 12-month price guarantee is the most "
                    f"effective lever now.")
        return (f"Approaching the 12-month mark on month-to-month. Trigger a contract upgrade "
                f"offer now — a modest discount (10–15%) with a price guarantee converts "
                f"flexibility into committed tenure without requiring a service upgrade.")

    if cluster_id == 3:  # High-Value Long-Term Anchors
        if contract == 'Month-to-month':
            return (f"High lifetime-value customer on month-to-month — the segment's highest "
                    f"risk profile. Prioritise a contract lock-in offer immediately: a "
                    f"price guarantee for 24 months protects significant revenue at risk.")
        if tenure > 60:
            return (f"Exceptional loyalty at {int(tenure)} months with ${charges:.0f}/mo spend. "
                    f"Proactive renewal outreach 60 days before contract end is essential — "
                    f"a complimentary add-on or rate guarantee signals that long-term "
                    f"customers are valued.")
        return (f"Long-tenure, high-value customer on a committed contract. Primary risk is "
                f"passive churn at renewal, not active dissatisfaction. Reach out 60 days "
                f"before renewal with a loyalty reward to ensure the next cycle.")

    if cluster_id == 4:  # Established Premium Loyalists
        if contract == 'Two year':
            return (f"Already on a two-year contract — maximum commitment. Focus on renewal "
                    f"timing: a service check-in 90 days before renewal and a loyalty rate "
                    f"guarantee maintains this high-value customer through the next cycle.")
        if contract == 'Month-to-month':
            return (f"${charges:.0f}/mo at {int(tenure)} months on month-to-month — atypical "
                    f"for this segment and elevated risk. An immediate two-year contract offer "
                    f"with a 5% discount secures approximately $2,000 of additional guaranteed "
                    f"lifetime value.")
        return (f"Proven long-term loyalty ({int(tenure)} months) on a one-year contract. "
                f"Migrate to two-year at renewal with a modest incentive (5% discount or "
                f"free add-on) — eliminates the annual renewal decision and locks in "
                f"approximately $2,000 of additional guaranteed lifetime value.")

    return "Review customer profile and apply segment-appropriate retention offer."


def build_why_here(cluster_id: int, data: dict) -> str:
    profile = cluster_profiles.get(str(cluster_id), {})
    tenure = float(data.get('tenure', 0))
    charges = float(data.get('MonthlyCharges', 0))
    contract = data.get('Contract', '')
    internet = data.get('InternetService', '')

    centroid_tenure = profile.get('mean_tenure', 0)
    centroid_charges = profile.get('mean_monthly_charges', 0)
    common_contract = profile.get('most_common_contract', '')
    common_internet = profile.get('most_common_internet', '')

    internet_label = {
        'Fiber optic': 'Fiber optic service',
        'DSL': 'DSL service',
        'No': 'no internet service'
    }.get(internet, internet)

    # Score each feature by closeness to cluster centroid — lower is better
    scored = [
        (abs(tenure - centroid_tenure) / 25,   f"{int(tenure)}-month tenure"),
        (abs(charges - centroid_charges) / 30,  f"${charges:.0f}/mo spend"),
        (0.0 if contract == common_contract else 1.0, f"{contract.lower()} contract"),
        (0.0 if internet == common_internet else 0.9,  internet_label),
    ]
    scored.sort(key=lambda x: x[0])

    labels = [s[1] for s in scored if s[0] < 0.65]
    if len(labels) < 2:
        labels = [s[1] for s in scored[:2]]
    labels = labels[:3]

    if len(labels) == 1:
        return f"Placed here by your {labels[0]}."
    if len(labels) == 2:
        return f"Placed here by your {labels[0]} and {labels[1]}."
    return f"Placed here by your {labels[0]}, {labels[1]}, and {labels[2]}."


def preprocess_input(data: dict) -> np.ndarray:
    """
    Apply the same preprocessing pipeline as train.py to a single
    customer record. Returns a scaled numpy array ready for prediction.
    """
    binary_map = {'Yes': 1, 'No': 0, 'No phone service': 0, 'No internet service': 0}

    contract = data.get('Contract', 'Month-to-month')
    internet = data.get('InternetService', 'Fiber optic')
    payment = data.get('PaymentMethod', 'Electronic check')

    # Manual one-hot encoding using the same reference categories as train.py
    # (drop_first=True drops the alphabetically first category in each group).
    # Reference categories: Contract=Month-to-month, InternetService=DSL,
    # PaymentMethod=Bank transfer (automatic).
    # pd.get_dummies on a single row can't infer the full category set, so
    # we encode explicitly to guarantee correct column alignment.
    row = {
        'tenure': float(data.get('tenure', 0)),
        'MonthlyCharges': float(data.get('MonthlyCharges', 0)),
        'TotalCharges': float(data.get('TotalCharges',
                              float(data.get('tenure', 0)) * float(data.get('MonthlyCharges', 0)))),
        'SeniorCitizen': int(data.get('SeniorCitizen', 0)),
        'TechSupport': binary_map.get(data.get('TechSupport', 'No'), 0),
        'OnlineSecurity': binary_map.get(data.get('OnlineSecurity', 'No'), 0),
        'PaperlessBilling': binary_map.get(data.get('PaperlessBilling', 'No'), 0),
        'Contract_One year':                    int(contract == 'One year'),
        'Contract_Two year':                    int(contract == 'Two year'),
        'InternetService_Fiber optic':          int(internet == 'Fiber optic'),
        'InternetService_No':                   int(internet == 'No'),
        'PaymentMethod_Credit card (automatic)': int(payment == 'Credit card (automatic)'),
        'PaymentMethod_Electronic check':        int(payment == 'Electronic check'),
        'PaymentMethod_Mailed check':            int(payment == 'Mailed check'),
    }

    df_row = pd.DataFrame([row])[feature_names]
    return scaler.transform(df_row)


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/clusters-3d')
def clusters_3d():
    return send_file(Path('plots/08_clusters_3d.html'))


@app.route('/predict', methods=['POST'])
def predict():
    if not MODEL_LOADED:
        return jsonify({'error': 'Models not loaded. Run train.py first.'}), 503

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON body received.'}), 400

    required_fields = ['tenure', 'MonthlyCharges', 'Contract', 'InternetService']
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({'error': f'Missing required fields: {missing}'}), 400

    X = preprocess_input(data)
    cluster_id = int(kmeans_model.predict(X)[0])
    pca_coords = pca_model.transform(X)[0]
    profile = get_segment_profile(cluster_id)

    cluster_summary = {
        cid: {
            'size': cp['size'],
            'tenure': cp['mean_tenure'],
            'monthly_charges': cp['mean_monthly_charges'],
            'churn_rate': cp['churn_rate']
        }
        for cid, cp in cluster_profiles.items()
    }

    return jsonify({
        'cluster_id': cluster_id,
        'segment_name': profile['name'],
        'segment_label': profile['label'],
        'description': profile['description'],
        'churn_risk': profile['churn_risk'],
        'retention_strategy': build_retention_strategy(cluster_id, data),
        'why_here': build_why_here(cluster_id, data),
        'colour': profile['colour'],
        'pca_x': float(pca_coords[0]),
        'pca_y': float(pca_coords[1]),
        'cluster_profiles': cluster_summary
    })


@app.route('/api/segments')
def api_segments():
    merged = {}
    for cluster_id, profile in SEGMENT_PROFILES.items():
        entry = dict(profile)
        cp = cluster_profiles.get(str(cluster_id), {})
        entry['size'] = cp.get('size')
        entry['pct_total'] = cp.get('pct_total')
        entry['churn_rate'] = cp.get('churn_rate')
        entry['mean_tenure'] = cp.get('mean_tenure')
        entry['mean_monthly_charges'] = cp.get('mean_monthly_charges')
        merged[cluster_id] = entry
    return jsonify(merged)


@app.route('/api/cluster-data')
def api_cluster_data():
    """
    Return PCA 2D coordinates and cluster labels for up to 2,000 randomly
    sampled customers — used by the frontend scatter plot.
    """
    if not MODEL_LOADED:
        return jsonify({'error': 'Models not loaded.'}), 503

    try:
        df = pd.read_csv(MODEL_DIR / 'clustered_data.csv')
    except FileNotFoundError:
        return jsonify({'error': 'clustered_data.csv not found. Run train.py first.'}), 503

    sample = df.sample(n=min(2000, len(df)), random_state=42)

    cluster_feature_cols = [c for c in feature_names if c in sample.columns]
    binary_map = {'Yes': 1, 'No': 0, 'No phone service': 0, 'No internet service': 0}
    for col in ['TechSupport', 'OnlineSecurity', 'PaperlessBilling']:
        if col in sample.columns and sample[col].dtype == object:
            sample[col] = sample[col].map(binary_map).fillna(sample[col])

    # Reconstruct dummies for categorical columns present in raw form
    cat_cols_present = [c for c in ['Contract', 'InternetService', 'PaymentMethod']
                        if c in sample.columns]
    if cat_cols_present:
        sample = pd.get_dummies(sample, columns=cat_cols_present, drop_first=True)

    for col in feature_names:
        if col not in sample.columns:
            sample[col] = 0
    X_sample = scaler.transform(sample[feature_names])
    coords_2d = pca_model.transform(X_sample)[:, :2]

    points = [
        {
            'x': float(coords_2d[i, 0]),
            'y': float(coords_2d[i, 1]),
            'cluster': int(sample.iloc[i]['KMeans_Cluster'])
        }
        for i in range(len(sample))
    ]
    return jsonify({'points': points})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
