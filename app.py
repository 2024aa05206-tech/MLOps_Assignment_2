from flask import Flask, request, jsonify, g
import time  
import logging 
import torch
from torchvision import transforms
from PIL import Image
import io
from src.models.model_file import BaselineCNN

app = Flask(__name__)

# ---- basic logging config (console) ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("inference_service")

# ---- simple in-memory metrics ----
request_count = 0
total_latency_ms = 0.0


# ------------ M5: monitoring hooks -------------
@app.before_request
def start_timer():
    """Store request start time per request."""
    g.start_time = time.time()


@app.after_request
def log_request(response):
    """
    After each request, compute latency, update counters,
    and log method, path, status, and latency.
    """
    global request_count, total_latency_ms

    if hasattr(g, "start_time"):
        latency_ms = (time.time() - g.start_time) * 1000
    else:
        latency_ms = 0.0

    request_count += 1
    total_latency_ms += latency_ms

    logger.info(
        f"method={request.method} path={request.path} "
        f"status={response.status_code} latency_ms={latency_ms:.2f}"
    )

    return response
# --------- end M5 monitoring hooks ------------

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
    return jsonify({"status": "ok", "version": "v2.0"}), 200

# --- M5: metrics endpoint ---
@app.route("/metrics", methods=["GET"])
def metrics():
    avg_latency = total_latency_ms / request_count if request_count > 0 else 0.0
    return jsonify({
        "request_count": request_count,
        "avg_latency_ms": avg_latency
    }), 200
# -------- end metrics endpoint ---------


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
    app.run(host="0.0.0.0", port=8000)
