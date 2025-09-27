from flask import Flask, request, jsonify
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load the trained model from the shared volume
MODEL_PATH = "/app/models/iris_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. Did the training container run and save it?"
    )
model = joblib.load(MODEL_PATH)

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json(silent=True)
    if not payload or 'input' not in payload:
        return jsonify({"error": "Request JSON must include 'input' array"}), 400

    iris_input = payload['input']
    if not isinstance(iris_input, (list, tuple)):
        return jsonify({"error": "'input' must be a list of numbers"}), 400

    try:
        arr = np.array(iris_input, dtype=float).reshape(1, -1)
    except Exception:
        return jsonify({"error": "'input' must be convertible to numeric array"}), 400

    try:
        prediction = model.predict(arr)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    return jsonify({"prediction": prediction[0]})

@app.route('/')
def hello():
    return 'Welcome to Docker Lab'

if __name__ == '__main__':
    #Run the Flask app (bind it to port 8080 or any other port)
    app.run(debug=True, port=8080, host='0.0.0.0')
