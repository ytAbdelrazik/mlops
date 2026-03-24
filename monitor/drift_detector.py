import json
from typing import List

import numpy as np
from scipy.stats import ks_2samp


def detect_drift(reference: List[float], current: List[float], threshold: float = 0.05) -> dict:
    stat, p_value = ks_2samp(reference, current)
    drift_detected = p_value < threshold

    return {
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p_value), 4),
        "drift_detected": drift_detected,
        "threshold": threshold,
    }


def load_scores(filepath: str) -> List[float]:
    with open(filepath) as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, help="Path to reference scores JSON")
    parser.add_argument("--current", required=True, help="Path to current scores JSON")
    parser.add_argument("--threshold", type=float, default=0.05)
    args = parser.parse_args()

    reference_scores = load_scores(args.reference)
    current_scores = load_scores(args.current)

    result = detect_drift(reference_scores, current_scores, args.threshold)
    print(json.dumps(result, indent=2))

    if result["drift_detected"]:
        print("WARNING: Data drift detected — consider retraining the model.")
