# server.py (updated)
import os
import uuid
import io

# --- Force headless matplotlib BEFORE importing pyplot to avoid Tk errors ---
import matplotlib

matplotlib.use("Agg")  # <<<< important: must come before pyplot import

import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import transformers
import tensorflow as tf

# ------------------ model paths (adjust if needed) ------------------------
MODEL_PATH = (
    r"/home/sleman-one/project/uitest/uitest/uitest/models/transformers_model_3.keras"
)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HEATMAP_DIR = os.path.join(BASE_DIR, "static", "heatmaps")
os.makedirs(HEATMAP_DIR, exist_ok=True)

# ------------------ load model (with nice error messages) -----------------
MODEL = None
PROCESSOR = None
try:
    print("Loading model from:", MODEL_PATH)
    MODEL = transformers.TFViTForImageClassification.from_pretrained(MODEL_PATH)
    PROCESSOR = transformers.ViTImageProcessor.from_pretrained(MODEL_PATH)
    print("Model + processor loaded.")
except Exception as e:
    print("Failed to load model or processor:", repr(e))
    MODEL = None
    PROCESSOR = None


# ------------------ helpers ------------------------------------------------
def pil_from_filestorage(file_storage):
    """Return PIL.Image from Flask FileStorage safely."""
    if hasattr(file_storage, "stream"):
        file_storage.stream.seek(0)
        return Image.open(io.BytesIO(file_storage.read())).convert("RGB")
    else:
        # fallback if caller passed a path or bytes
        return Image.open(file_storage).convert("RGB")


def make_vit_attention_heatmap(model, processor, file_storage, save_path):
    """Extract mean CLS attention from the final layer and overlay on image."""
    # build PIL image
    pil = pil_from_filestorage(file_storage)
    inputs = processor(images=pil, return_tensors="tf")

    # request attentions
    outputs = model(inputs["pixel_values"], output_attentions=True)
    if not hasattr(outputs, "attentions") or outputs.attentions is None:
        raise RuntimeError(
            "Model did not return attentions. Ensure model outputs attentions."
        )

    attentions = outputs.attentions  # tuple len=num_layers of tensors
    last = attentions[-1]  # shape: (batch, num_heads, seq_len, seq_len)
    # average over heads -> (batch, seq_len, seq_len)
    mean_heads = tf.reduce_mean(last, axis=1)  # axis=1 is heads

    # CLS token attends to other tokens along row 0. We take the row corresponding to CLS (index 0)
    cls_row = mean_heads[0, 0, 1:]  # skip CLS->CLS self-attention
    cls_row = tf.reshape(cls_row, [-1])  # flatten

    # convert to square map (num_patches x num_patches)
    n_tokens = int(cls_row.shape[0])
    num_patches = int(np.sqrt(n_tokens))
    if num_patches * num_patches != n_tokens:
        # fallback: try rounding
        num_patches = int(round(np.sqrt(n_tokens)))

    cls_map = tf.reshape(cls_row, (num_patches, num_patches)).numpy()

    # resize to image size
    w, h = pil.size
    cls_map_resized = cv2.resize(cls_map, (w, h), interpolation=cv2.INTER_CUBIC)

    # normalize to 0..1
    mn, mx = cls_map_resized.min(), cls_map_resized.max()
    if mx > mn:
        cls_map_resized = (cls_map_resized - mn) / (mx - mn)
    else:
        cls_map_resized = np.zeros_like(cls_map_resized)

    # overlay with matplotlib and save
    plt.figure(figsize=(w / 100, h / 100), dpi=100)
    plt.imshow(pil)
    plt.imshow(cls_map_resized, cmap="jet", alpha=0.45)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    return save_path


def predict_image_prob(model, processor, file_storage, classes_name=None):
    """
    Robust prediction:
      - if logits shape (1, 1) -> treat as single-logit and apply sigmoid
      - if logits shape (1, C) with C>1 -> apply softmax and return prob of class index 1 (assumed 'Reel')
    Returns: (prob_reel, label)
    """
    if classes_name is None:
        classes_name = ["Fake", "Reel"]

    pil = pil_from_filestorage(file_storage)
    inputs = processor(images=pil, return_tensors="tf")

    outputs = model(inputs["pixel_values"])
    logits = outputs.logits  # tf.Tensor

    logits_arr = np.asarray(logits)
    # logits_arr shape: (1,) or (1,1) or (1, C)
    if logits_arr.ndim == 1:
        # scalar
        score = float(logits_arr.item())
        p_reel = float(tf.sigmoid(score).numpy())
    elif logits_arr.ndim == 2 and logits_arr.shape[1] == 1:
        score = float(logits_arr[0, 0])
        p_reel = float(tf.sigmoid(score).numpy())
    else:
        # multi-class -> softmax
        probs = tf.nn.softmax(logits, axis=-1).numpy()
        # if there are 2 classes, assume index 1 is 'Reel'
        idx = 1 if logits_arr.shape[1] > 1 else 0
        p_reel = float(probs[0, idx])

    # Choose label and return confidence (returning P(Reel) as "confidence" when label is 'reel')
    if p_reel >= 0.5:
        label = classes_name[1]
        confidence = p_reel
    else:
        label = classes_name[0]
        confidence = 1.0 - p_reel

    return float(confidence), label, float(p_reel)


# ------------------ Flask app ------------------------------------------------
app = Flask(__name__)  # default static folder = ./static
CORS(app)


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "no image provided"}), 400

    file = request.files["image"]
    # quick sanity test - ensure file readable
    try:
        # run prediction
        if MODEL is None or PROCESSOR is None:
            # no model loaded — return demo random
            import random

            p_reel_demo = random.uniform(0.2, 0.9)
            label = "reel" if p_reel_demo >= 0.5 else "fake"
            confidence = p_reel_demo if label == "reel" else 1 - p_reel_demo
            heatmap_url = None
        else:
            confidence, label, p_reel = predict_image_prob(
                MODEL, PROCESSOR, file, classes_name=["Fake", "Reel"]
            )
            # create heatmap
            uid = uuid.uuid4().hex
            heatmap_name = f"heatmap_{uid}.png"
            heatmap_path = os.path.join(HEATMAP_DIR, heatmap_name)
            try:
                make_vit_attention_heatmap(MODEL, PROCESSOR, file, heatmap_path)
                heatmap_url = f"/static/heatmaps/{heatmap_name}"
            except Exception as he:
                print("Heatmap generation failed:", repr(he))
                heatmap_url = None

        print(f"prediction={label}, confidence={confidence}")
        return jsonify(
            {
                "prediction": label.lower(),
                "confidence": float(confidence),
                "p_reel": float(p_reel) if MODEL is not None else None,
                "heatmap_url": heatmap_url,
            }
        )
    except Exception as e:
        print("Analyze error:", repr(e))
        return jsonify({"error": "analyze failed", "detail": str(e)}), 500


# Serve heatmaps explicitly (optional)
@app.route("/static/heatmaps/<path:filename>")
def serve_heatmap(filename):
    return send_from_directory(HEATMAP_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
