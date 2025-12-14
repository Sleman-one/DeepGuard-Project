import os
import io
import base64  # <--- NEW IMPORT
import matplotlib

matplotlib.use("Agg")  # Force headless backend
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import transformers
import tensorflow as tf

# ------------------ Paths ------------------
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "models", "transformers_model_3.keras"
)

# ------------------ Load Model ------------------
MODEL = None
PROCESSOR = None
try:
    print("⏳ Loading model...")
    MODEL = transformers.TFViTForImageClassification.from_pretrained(MODEL_PATH)
    PROCESSOR = transformers.ViTImageProcessor.from_pretrained(MODEL_PATH)
    print("✅ Model loaded!")
except Exception as e:
    print("❌ Failed to load model:", repr(e))


# ------------------ Helper Functions ------------------
def pil_from_filestorage(file_storage):
    """Convert upload to PIL Image safely."""
    if hasattr(file_storage, "stream"):
        file_storage.stream.seek(0)
        return Image.open(io.BytesIO(file_storage.read())).convert("RGB")
    return Image.open(file_storage).convert("RGB")


def make_heatmap_base64(model, processor, pil_img):
    """Generate heatmap and return as Base64 string (No saving to disk)."""
    inputs = processor(images=pil_img, return_tensors="tf")
    outputs = model(inputs["pixel_values"], output_attentions=True)

    if not hasattr(outputs, "attentions") or outputs.attentions is None:
        return None

    # Get attention map
    last_layer = outputs.attentions[-1]
    mean_heads = tf.reduce_mean(last_layer, axis=1)
    cls_row = mean_heads[0, 0, 1:]

    # Reshape to square
    n_tokens = int(cls_row.shape[0])
    side = int(np.sqrt(n_tokens))
    cls_map = tf.reshape(cls_row, (side, side)).numpy()

    # Resize to match original image
    w, h = pil_img.size
    cls_map_resized = cv2.resize(cls_map, (w, h), interpolation=cv2.INTER_CUBIC)

    # Normalize 0..1
    mn, mx = cls_map_resized.min(), cls_map_resized.max()
    if mx > mn:
        cls_map_resized = (cls_map_resized - mn) / (mx - mn)

    # Create plot in memory
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(pil_img)
    ax.imshow(cls_map_resized, cmap="jet", alpha=0.45)

    # Save to RAM (Buffer)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # Convert to Base64 String
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"


def predict(model, processor, pil_img):
    inputs = processor(images=pil_img, return_tensors="tf")
    logits = model(inputs["pixel_values"]).logits.numpy()

    # Calculate Probabilities
    if logits.ndim == 2 and logits.shape[1] > 1:
        probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
        p_real = float(probs[1])  # Assuming Index 1 is Real
    else:
        p_real = float(tf.sigmoid(logits).numpy().item())

    label = "real" if p_real >= 0.5 else "fake"
    conf = p_real if label == "real" else 1.0 - p_real
    return label, conf


# ------------------ Flask App ------------------
app = Flask(__name__)
CORS(app)


@app.route("/analyze", methods=["POST"])
def analyze_route():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    try:
        # 1. Load Image
        img = pil_from_filestorage(file)

        # 2. Predict
        if MODEL:
            label, conf = predict(MODEL, PROCESSOR, img)

            # 3. Generate Heatmap (Direct Base64)
            heatmap_b64 = make_heatmap_base64(MODEL, PROCESSOR, img)

            return jsonify(
                {
                    "prediction": label,
                    "confidence": conf,
                    "heatmap": heatmap_b64,  # <--- Sending actual image data!
                }
            )
        else:
            return jsonify({"error": "Model not loaded"}), 500

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
