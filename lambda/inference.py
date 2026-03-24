import boto3, pickle, io, json, logging
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client('s3')
BUCKET = 'spam-classifier-mlops-yt2'
MODEL_KEY = 'models/naive_bayes_v1.pkl'

model = None

def load_model():
    global model
    if model is None:
        obj = s3.get_object(Bucket=BUCKET, Key=MODEL_KEY)
        model = pickle.load(io.BytesIO(obj['Body'].read()))
        logger.info('Model loaded!')
    return model

def lambda_handler(event, context):
    try:
        body = json.loads(event.get('body', '{}'))
        text = body.get('text', '').strip()

        if not text:
            return {'statusCode': 400,
                    'body': json.dumps({'error': 'No text provided'})}

        clf = load_model()
        prediction = clf.predict([text])[0]
        confidence = round(float(clf.predict_proba([text]).max()), 4)

        logger.info(json.dumps({
            'timestamp':  datetime.utcnow().isoformat(),
            'prediction': prediction,
            'confidence': confidence
        }))

        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'prediction': prediction,
                'confidence': confidence,
                'is_spam':    prediction == 'spam'
            })
        }

    except Exception as e:
        logger.error(str(e))
        return {'statusCode': 500,
                'body': json.dumps({'error': str(e)})}