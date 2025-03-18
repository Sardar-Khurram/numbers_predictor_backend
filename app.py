from flask import Flask, request, jsonify
import joblib
import os
from flask_cors import CORS

app = Flask(__name__)

# ✅ Correct CORS configuration
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Load the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_mlp_model.pkl")
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route("/", methods=["GET"])
def home():
    return "Flask Backend is Running!"

# ✅ Remove unnecessary OPTIONS route (Flask-CORS handles it)
@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "Invalid input. 'image' field is required."}), 400

        image = data["image"]
        if not isinstance(image, list) or len(image) != 784:
            return jsonify({"error": "Invalid input. Expected a flattened array of 784 pixel values."}), 400

        # Count white & black pixels
        white_pixels = sum(1 for pixel in image if pixel > 200)  # Almost white (200-255)
        black_pixels = sum(1 for pixel in image if pixel < 50)   # Almost black (0-50)

        white_ratio = white_pixels / 784
        black_ratio = black_pixels / 784

        print(f"White ratio: {white_ratio:.2%}, Black ratio: {black_ratio:.2%}")  # Debugging

        # 🛑 Check if the image is completely empty (all white)
        if white_pixels == 784:
            return jsonify({"error": "Image is required. It cannot be empty."}), 400

        # 🛑 Ensure the image actually looks like a handwritten number
        if white_ratio < 0.60 or black_ratio < 0.05:
            return jsonify({"error": "This does not appear to be a handwritten number."}), 400

        # ✅ Make prediction
        prediction = model.predict([image])
        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "Invalid input. 'image' field is required."}), 400

        image = data["image"]
        if not isinstance(image, list) or len(image) != 784:
            return jsonify({"error": "Invalid input. Expected a flattened array of 784 pixel values."}), 400

        prediction = model.predict([image])
        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    PORT = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=PORT)
