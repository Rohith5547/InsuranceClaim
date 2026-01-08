from flask import Flask, request, render_template
import pandas as pd
import base64
import io
import requests

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # or just return a simple string
    # return "Welcome to the Insurance Claim Prediction App"


@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded file data (base64 encoded)
    file_data = request.form.get('file')
    if not file_data:
        return "No file provided", 400

    # Decode the CSV file
    decoded_file = base64.b64decode(file_data.split(',')[1])
    df = pd.read_csv(io.StringIO(decoded_file.decode('utf-8')))

    # Safely extract claim_id if it exists
    claim_ids = df['claim_id'] if 'claim_id' in df.columns else None
    if 'claim_id' in df.columns:
        df = df.drop(columns=['claim_id'])

    # Expected features for prediction
    expected_features = [
        'claim_amount',
        'num_services',
        'patient_age',
        'provider_id',
        'days_since_last_claim'
    ]

    # Keep only expected features
    df = df[expected_features]

    # Fill missing numeric values
    df = df.fillna(df.mean(numeric_only=True))

    # Call the BentoML prediction API
    try:
        response = requests.post(
            'http://127.0.0.1:3000/predict',
            json={"data": df.to_dict(orient='list')},  # <-- wrap inside "data"
            timeout=5
        )

    except requests.exceptions.RequestException as e:
        return f"Error connecting to prediction service: {str(e)}", 500

    if not response.ok:
        return f"BentoML error: {response.text}", 500

    # Get predictions and add them to the DataFrame
    data = response.json()
    df['Prediction'] = data.get('predictions', [])

    # Add claim_id back if it existed
    if claim_ids is not None:
        df['claim_id'] = claim_ids

    # Render results
    return render_template(
        'results.html',
        tables=[df.to_html(classes='data', index=False)]
    )

if __name__ == '__main__':
    app.run(port=5005)