import os
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# Enable Cross-Origin Resource Sharing (CORS)
CORS(app)

# Load the pre-trained model
MODEL_PATH = "model.joblib"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
else:
    raise FileNotFoundError(f"Could not find {MODEL_PATH}. Make sure it is in the same folder.")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No input data provided"}), 400

        ref = data.get("ref", "A").upper().strip()
        alt = data.get("alt", "G").upper().strip()

        ref_len = len(ref)
        alt_len = len(alt)

        if ref_len == 1 and alt_len == 1:
            var_type = 0
        elif ref_len > alt_len:
            var_type = 1
        elif ref_len < alt_len:
            var_type = 2
        else:
            var_type = 3

        input_features = [[ref_len, alt_len, var_type]]

        prediction = int(model.predict(input_features)[0])
        probabilities = model.predict_proba(input_features)[0]
        confidence = float(probabilities[prediction]) * 100

        mutation_types = ["SNV", "Deletion", "Insertion", "Complex"]
        result_label = "Pathogenic" if prediction == 1 else "Benign"

        return jsonify({
            "status": "success",
            "prediction": result_label,
            "confidence": f"{confidence:.2f}%",
            "type": mutation_types[var_type]
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
