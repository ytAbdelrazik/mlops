# Spam Classifier MLOps

An end-to-end MLOps pipeline for training, deploying, and monitoring a spam classification model on AWS.

## Project Structure

```
mlops/
├── data/
│   └── prepare_data.py     # Load, preprocess, and split raw data
├── train/
│   └── train.py            # Train TF-IDF + Logistic Regression pipeline
├── lambda/
│   └── inference.py        # AWS Lambda handler for real-time inference
├── monitor/
│   └── drift_detector.py   # KS-test based data drift detection
├── buildspec.yml           # AWS CodeBuild CI/CD configuration
├── requirements.txt        # Python dependencies
└── README.md
```

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare data
```bash
python data/prepare_data.py
```
Expects `data/raw.csv` with `text` and `label` columns.

### 3. Train the model
```bash
python train/train.py --train-path data/train.csv --model-output model/model.pkl
```

### 4. Detect drift
```bash
python monitor/drift_detector.py \
  --reference data/reference_scores.json \
  --current data/current_scores.json
```

### 5. Deploy via CodeBuild
Push to your repository — CodeBuild will run `buildspec.yml` automatically to retrain and upload the model to S3.

## Environment Variables

| Variable       | Description                        |
|----------------|------------------------------------|
| `MODEL_BUCKET` | S3 bucket where model is stored    |
| `MODEL_KEY`    | S3 key for the model artifact      |

## Lambda Inference

Send a POST request with:
```json
{ "text": "Congratulations! You won a free prize." }
```

Response:
```json
{ "prediction": "spam", "confidence": 0.97 }
```
