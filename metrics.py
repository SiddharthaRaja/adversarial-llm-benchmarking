import pandas as pd


def compute_metrics(df):

    total = len(df)

    classification_counts = (
        df["classification"]
        .value_counts()
        .to_dict()
    )

    average_risk = round(
        df["risk_score"].mean(),
        3
    )

    metrics = {
        "total_attacks": total,
        "average_risk_score": average_risk
    }

    for label, count in classification_counts.items():

        metrics[f"{label}_rate"] = round(
            (count / total) * 100,
            2
        )

    return metrics
