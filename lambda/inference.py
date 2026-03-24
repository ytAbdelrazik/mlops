import json
import os
import pickle
import boto3


MODEL_BUCKET = os.environ.get("MODEL_BUCKET", "")
MODEL_KEY = os.environ.get("MODEL_KEY", "model/model.pkl")

_model = None


def load_model():
    global _model
    if _model is None:
        s3 = boto3.client("s3")
        tmp_path = "/tmp/model.pkl"
        s3.download_file(MODEL_BUCKET, MODEL_KEY, tmp_path)
        with open(tmp_path, "rb") as f:
            _model = pickle.load(f)
    return _model


def handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))
        text = body.get("text", "")

        if not text:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing 'text' field"}),
            }

        model = load_model()
        prediction = model.predict([text])[0]
        probability = model.predict_proba([text])[0].max()

        return {
            "statusCode": 200,
            "body": json.dumps({
                "prediction": str(prediction),
                "confidence": round(float(probability), 4),
            }),
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
        }
