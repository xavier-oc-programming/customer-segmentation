# segment_labels.py
# Business labels for customer segments identified by KMeans.
# IMPORTANT: Fill in the TBD values after running train.py and
# inspecting the cluster profiles printed in the final summary.
# The cluster numbering (0-4) is assigned by KMeans arbitrarily —
# cluster 0 in one run may not be cluster 0 in the next.
# After filling in labels, re-run app.py to reflect updated profiles.

SEGMENT_PROFILES = {
    0: {
        "name": "At-Risk New Adopters",
        "label": "At-Risk",
        "description": (
            "Customers who tend to be earlier in their relationship with the company, often on "
            "flexible contracts and paying above-average prices. Without a strong sense of loyalty "
            "established, the connection can feel transactional — and this group churns at over "
            "twice the dataset average. The window to build that loyalty is typically narrow."
        ),
        "churn_risk": "high",
        "retention_strategy": (
            "Intervene within the first 90 days. Offer a price-locked annual contract bundled with one "
            "add-on service (TechSupport or OnlineSecurity). A proactive onboarding call at 30 days reduces "
            "early churn in this profile — the goal is to build perceived value before the cancellation "
            "window opens."
        ),
        "colour": "#E74C3C"
    },
    1: {
        "name": "Stable Budget Loyalists",
        "label": "Stable",
        "description": (
            "Among the company's most dependable customer profiles. Typically on long-term contracts "
            "with modest spend, this group tends to stay put — not out of high engagement, but out of "
            "inertia and satisfaction with a service that meets their needs. Generally low maintenance, "
            "low risk, and a reliable source of predictable revenue."
        ),
        "churn_risk": "low",
        "retention_strategy": (
            "Retention spend here has low ROI — this segment is not at risk. The opportunity is upsell: "
            "these customers currently pay $21/mo against a dataset average of $65. A targeted DSL or "
            "Fiber optic introduction offer bundled with a contract extension could significantly increase "
            "their lifetime value without disrupting their loyalty."
        ),
        "colour": "#3498DB"
    },
    2: {
        "name": "Uncommitted New Subscribers",
        "label": "Uncommitted",
        "description": (
            "Customers who tend to be earlier in their tenure and often on flexible contracts, "
            "suggesting a relationship that is still forming. This profile can be sensitive to "
            "competitor offers or pricing changes, not necessarily out of dissatisfaction, but "
            "because no strong reason to stay has taken hold yet."
        ),
        "churn_risk": "medium",
        "retention_strategy": (
            "Target a contract upgrade at the 12-month mark before the churn window fully opens. DSL "
            "customers in this profile are price-sensitive — a modest discount (10–15%) with a price "
            "guarantee for 12 months converts month-to-month flexibility into committed tenure without "
            "requiring a full service upgrade."
        ),
        "colour": "#2ECC71"
    },
    3: {
        "name": "High-Value Long-Term Anchors",
        "label": "Anchor",
        "description": (
            "Among the highest lifetime-value profiles in the dataset. Typically long-tenured and "
            "on committed contracts, this group tends to stay because the relationship is working — "
            "not because they're locked in. Churn risk is minimal, but a single loss in this group "
            "can represent a significant revenue event."
        ),
        "churn_risk": "low",
        "retention_strategy": (
            "Protect and reward. Reach out proactively 60 days before contract renewal with a loyalty "
            "offer — a complimentary add-on or modest rate guarantee signals that long-term customers are "
            "valued. Passive churn at renewal (forgetting to renew rather than active dissatisfaction) is "
            "the primary risk here."
        ),
        "colour": "#F39C12"
    },
    4: {
        "name": "Established Premium Loyalists",
        "label": "Power User",
        "description": (
            "A well-established customer profile — typically longer tenure, higher spend, and a "
            "track record of staying. Often sitting on shorter-term contracts despite years of "
            "loyalty, this group can be one renewal cycle away from a lapse. Generally strong "
            "candidates for a longer-term commitment if approached at the right moment."
        ),
        "churn_risk": "low",
        "retention_strategy": (
            "Migrate from one-year to two-year contracts at renewal with a modest incentive (5% rate "
            "discount or free add-on). These customers have demonstrated 4+ years of Fiber optic loyalty — "
            "a two-year lock-in eliminates the annual renewal churn risk and secures approximately $2,000 "
            "of additional guaranteed lifetime value per converted customer."
        ),
        "colour": "#9B59B6"
    }
}


def get_segment_profile(cluster_id: int) -> dict:
    """
    Return the business profile for a given cluster ID.

    Args:
        cluster_id: integer cluster label from KMeans

    Returns:
        Dict with keys: name, label, description, churn_risk,
        retention_strategy, colour.
        Returns a default "Unknown segment" dict if cluster_id
        not found.
    """
    return SEGMENT_PROFILES.get(cluster_id, {
        "name": "Unknown segment",
        "label": "Unknown",
        "description": "No profile available for this cluster.",
        "churn_risk": "unknown",
        "retention_strategy": "No strategy defined.",
        "colour": "#95A5A6"
    })
