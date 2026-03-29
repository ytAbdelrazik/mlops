import boto3, joblib, io, json
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from datetime import datetime

BUCKET = 'spam-classifier-mlops-YOUR_NAME'
s3 = boto3.client('s3')
cloudwatch = boto3.client('cloudwatch')


def load_model(version='v1'):
    obj = s3.get_object(Bucket=BUCKET, Key=f'models/naive_bayes_{version}.pkl')
    return joblib.load(io.BytesIO(obj['Body'].read()))


def load_baseline_stats():
    obj = s3.get_object(Bucket=BUCKET, Key='models/baseline_stats_v1.json')
    return json.loads(obj['Body'].read())


def get_confidence_scores(model, texts):
    return model.predict_proba(list(texts)).max(axis=1)


def run_drift_check(batch_key, model_version='v1'):
    model = load_model(model_version)
    baseline = load_baseline_stats()

    # Load new batch
    obj = s3.get_object(Bucket=BUCKET, Key=batch_key)
    df = pd.read_csv(io.BytesIO(obj['Body'].read())).dropna()

    # Get confidence scores for new batch
    new_scores = get_confidence_scores(model, df['text'])
    new_mean = float(np.mean(new_scores))

    # KS test: compare new scores to a normal distribution
    # centered on baseline mean
    baseline_scores = np.random.normal(
        baseline['mean_confidence'],
        baseline['std_confidence'],
        len(new_scores)
    )
    stat, p_value = ks_2samp(baseline_scores, new_scores)
    drift_detected = p_value < 0.05 or new_mean < 0.80

    result = {
        'timestamp': datetime.utcnow().isoformat(),
        'batch': batch_key,
        'samples_checked': len(new_scores),
        'new_mean_conf': round(new_mean, 4),
        'baseline_mean': round(baseline['mean_confidence'], 4),
        'confidence_drop': round(baseline['mean_confidence'] - new_mean, 4),
        'ks_statistic': round(stat, 4),
        'p_value': round(p_value, 6),
        'drift_detected': drift_detected
    }
    print(json.dumps(result, indent=2))

    # Push metric to CloudWatch
    cloudwatch.put_metric_data(
        Namespace='SpamClassifier',
        MetricData=[
            {'MetricName': 'BatchMeanConfidence',
             'Value': new_mean, 'Unit': 'None'},
            {'MetricName': 'DriftDetected',
             'Value': 1 if drift_detected else 0, 'Unit': 'Count'}
        ]
    )
    return result


if __name__ == '__main__':
    # Run on Enron batch
    r = run_drift_check('data/production_batches/batch_1_enron.csv')
    if r['drift_detected']:
        print('DRIFT DETECTED — model retraining recommended!')