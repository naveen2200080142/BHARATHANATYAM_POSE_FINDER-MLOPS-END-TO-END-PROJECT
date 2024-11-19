from flask import Flask, request, jsonify
import requests
import base64
import os

app = Flask(__name__)

# Set the correct Vertex AI endpoint URL (replace with actual API URL)
VERTEX_AI_ENDPOINT = os.getenv('VERTEX_AI_ENDPOINT')

@app.route('/predict', methods=['POST'])
def predict_pose():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    # Read the uploaded image file
    image = request.files['file'].read()

    # Encode the image to base64
    image_base64 = base64.b64encode(image).decode('utf-8')

    # Prepare the payload for the Vertex AI request
    payload = {
        "instances": [
            {"b64": image_base64}
        ]
    }

    # Set the headers for the Vertex AI request
    headers = {
        'Authorization': f'Bearer {os.getenv("GCP_ACCESS_TOKEN")}',  # Add your GCP access token
        'Content-Type': 'application/json',
    }

    # Send the request to Vertex AI endpoint
    response = requests.post(VERTEX_AI_ENDPOINT, json=payload, headers=headers)

    # Check the response status and return the result
    if response.status_code == 200:
        prediction = response.json()
        # Get the predicted pose (modify based on actual Vertex AI response structure)
        predicted_pose = prediction['predictions'][0]['pose']
        return jsonify({'predicted_pose': predicted_pose})
    else:
        return jsonify({'error': 'Failed to get prediction'}), 500

if __name__ == '__main__':
    app.run(debug=True)
