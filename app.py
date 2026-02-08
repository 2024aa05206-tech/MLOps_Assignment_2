from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io
from model_file import BaselineCNN

app = Flask(__name__)

# Load model once at startup
model = BaselineCNN()
model.load_state_dict(torch.load("models/baseline_cnn.pt", map_location="cpu"))
model.eval()

# Preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# --- Health check endpoint ---
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# --- Prediction endpoint ---
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = transform(img).unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        prob_dog = model(img).item()
        prob_cat = 1 - prob_dog
        label = "dog" if prob_dog > 0.5 else "cat"

    return jsonify({
        "label": label,
        "probabilities": {
            "cat": prob_cat,
            "dog": prob_dog
        }
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
