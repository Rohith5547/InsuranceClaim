from flask import Flask, render_template, request
import pandas as pd
import requests
import base64
import io


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    file_data = request.form.get('file')

    decoded_file = base64.b64decode(file_data.split(',')[1])
    df = pd.read_csv(io.StringIO(decoded_file.decode('utf-8')))

    # Separate claim_id
    if 'claim_id' in df.columns:
        claim_ids = df['claim_id']
        df = df.drop(columns=['claim_id'])
    else:
        claim_ids = None

    # 🔴 CRITICAL FIX 1: Replace NaN with None
    df = df.where(pd.notnull(df), None)

    # 🔴 CRITICAL FIX 2: Column-oriented JSON
    response = requests.post(
        'http://127.0.0.1:3000/predict',
        json=df.to_dict(orient='list'),
        timeout=5
    )

    if not response.ok:
        return f"BentoML error: {response.text}", 500

    data = response.json()

    if 'predictions' not in data:
        return f"Invalid model response: {data}", 500

    df['Prediction'] = data['predictions']

    if claim_ids is not None:
        df['claim_id'] = claim_ids

    return render_template(
        'results.html',
        tables=[df.to_html(classes='data', index=False)]
    )

    
if __name__ == '__main__':
    app.run(port=5005)