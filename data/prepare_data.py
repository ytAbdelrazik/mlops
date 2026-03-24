import pandas as pd
import boto3

BUCKET = 'spam-classifier-mlops-yt2'
s3 = boto3.client('s3')

# ── SMS Spam ──────────────────────────────────────────────────────
sms = pd.read_csv(
    'SMSSpamCollection',
    sep='\t',
    header=None,
    names=['label', 'text']
)

print('SMS shape:', sms.shape)
print(sms['label'].value_counts())

# ── Enron (emails.csv has no label → treat as ham) ────────────────
enron = pd.read_csv('emails.csv')[['message']].rename(columns={'message': 'text'})
enron['label'] = 'ham'
enron = enron.dropna().sample(3000, random_state=42)  # use 3K samples

# ── Lingspam ──────────────────────────────────────────────────────
lingspam = pd.read_csv('messages.csv')[['message', 'label']].rename(columns={'message': 'text'})
lingspam['label'] = lingspam['label'].map({1: 'spam', 0: 'ham'})
lingspam = lingspam.dropna()

# ── Save locally ──────────────────────────────────────────────────
sms.to_csv('sms_spam.csv', index=False)
enron.to_csv('batch_1_enron.csv', index=False)
lingspam.to_csv('batch_2_lingspam.csv', index=False)

# ── Upload to S3 ──────────────────────────────────────────────────
files = {
    'sms_spam.csv': 'data/train/sms_spam.csv',
    'batch_1_enron.csv': 'data/production_batches/batch_1_enron.csv',
    'batch_2_lingspam.csv': 'data/production_batches/batch_2_lingspam.csv'
}

for local, key in files.items():
    s3.upload_file(local, BUCKET, key)
    print(f'Uploaded {key}')

print('Done! All data in S3.')