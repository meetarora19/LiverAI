import os
import sys
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.preprocessing import image

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

MODEL_PATH = os.path.join(BASE_DIR, "models", "ct_model", "saved_models", "ct_classifier.keras")

OUTPUT_DIR = os.path.join(BASE_DIR, "data", "ct_liver", "outputs", "predictions")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ["Normal", "Tumor"]
IMG_SIZE = (224, 224)


def predict_ct(img_path: str):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    model = tf.keras.models.load_model(MODEL_PATH)

    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)[0]

    pred_idx = int(np.argmax(prediction))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(np.max(prediction)) * 100

    if pred_class == "Tumor":
        result = "Abnormality Detected"
    else:
        result = "No Major Abnormality Detected"

    # 🔥 Timestamp for unique saving
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    # 🔥 Save image with prediction
    output_img_path = os.path.join(OUTPUT_DIR, f"{base_name}_{timestamp}.png")

    plt.figure(figsize=(6, 6))
    plt.imshow(image.load_img(img_path))
    plt.title(
        f"Result: {result}\nPredicted Type: {pred_class}\nConfidence: {confidence:.1f}%",
        fontsize=12
    )
    plt.axis("off")
    plt.savefig(output_img_path, bbox_inches="tight")
    plt.close()

    # 🔥 Save JSON output
    output_json_path = os.path.join(OUTPUT_DIR, f"{base_name}_{timestamp}.json")

    result_data = {
        "image": img_path,
        "result": result,
        "predicted_type": pred_class,
        "confidence": round(confidence, 2),
        "raw_probabilities": prediction.tolist()
    }

    with open(output_json_path, "w") as f:
        json.dump(result_data, f, indent=4)

    # Console output
    print(f"\nResult: {result}")
    print(f"Predicted Type: {pred_class}")
    print(f"Confidence: {confidence:.1f}%")
    print("Saved Image:", output_img_path)
    print("Saved JSON:", output_json_path)

    return result_data


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/ct_pipeline/predict_ct.py <image_path>")
    else:
        predict_ct(sys.argv[1])