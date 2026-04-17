import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
import json
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.preprocessing import image

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

FATTY_MODEL_PATH = os.path.join(BASE_DIR, "models", "fatty_liver", "fatty_liver_model_224_finetuned.keras")
CT_MODEL_PATH = os.path.join(BASE_DIR, "models", "ct_model", "saved_models", "ct_classifier.keras")

OUTPUT_DIR = os.path.join(BASE_DIR, "data", "ct_liver", "outputs", "predictions")

LIVER_MASK_DIR = os.path.join(BASE_DIR, "data", "ct_liver", "processed", "masks", "liver")
TUMOR_MASK_DIR = os.path.join(BASE_DIR, "data", "ct_liver", "processed", "masks", "tumor")

IMG_SIZE = (224, 224)

FATTY_CLASSES = ["NAFLD", "Non-NAFLD"]
CT_CLASSES = ["Normal", "Tumor"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_models():
    fatty_model = tf.keras.models.load_model(FATTY_MODEL_PATH)
    ct_model = tf.keras.models.load_model(CT_MODEL_PATH)
    return fatty_model, ct_model


def preprocess(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return img, arr


def fatty_prediction(model, arr):
    pred = model.predict(arr, verbose=0)[0]

    nafld_idx = FATTY_CLASSES.index("NAFLD")
    non_idx = FATTY_CLASSES.index("Non-NAFLD")

    nafld_prob = float(pred[nafld_idx])
    non_prob = float(pred[non_idx])

    margin = abs(nafld_prob - non_prob)

    if margin < 0.15:
        result = "Uncertain"
        pred_class = FATTY_CLASSES[int(np.argmax(pred))]
        confidence = max(nafld_prob, non_prob) * 100
    elif nafld_prob > non_prob:
        result = "Fatty Liver Detected"
        pred_class = "NAFLD"
        confidence = nafld_prob * 100
    else:
        result = "No Fatty Liver Detected"
        pred_class = "Non-NAFLD"
        confidence = non_prob * 100

    return {
        "result": result,
        "type": pred_class,
        "confidence": round(confidence, 2)
    }


def ct_prediction(model, arr):
    pred = model.predict(arr, verbose=0)[0]

    idx = int(np.argmax(pred))
    pred_class = CT_CLASSES[idx]
    confidence = float(np.max(pred)) * 100

    result = "Abnormality Detected" if pred_class == "Tumor" else "No Major Abnormality"

    return {
        "result": result,
        "type": pred_class,
        "confidence": round(confidence, 2)
    }


def morphology(img_path):
    name = os.path.basename(img_path)

    liver_path = os.path.join(LIVER_MASK_DIR, name)
    tumor_path = os.path.join(TUMOR_MASK_DIR, name)

    if not os.path.exists(liver_path) or not os.path.exists(tumor_path):
        return None

    liver = cv2.imread(liver_path, 0)
    tumor = cv2.imread(tumor_path, 0)

    liver = (liver > 0).astype(np.uint8)
    tumor = (tumor > 0).astype(np.uint8)

    liver_area = int(liver.sum())
    tumor_area = int(tumor.sum())

    ratio = tumor_area / liver_area if liver_area > 0 else 0

    coords = cv2.findNonZero((tumor * 255).astype(np.uint8))
    if coords is None:
        w, h = 0, 0
    else:
        _, _, w, h = cv2.boundingRect(coords)

    return {
        "tumor_area": tumor_area,
        "liver_area": liver_area,
        "ratio": round(ratio, 6),
        "size_px": f"{w}x{h}"
    }


def final_decision(fatty, ct):
    abnormalities = []

    if fatty["result"] == "Fatty Liver Detected":
        abnormalities.append("Fatty Liver")

    if ct["type"] == "Tumor":
        abnormalities.append("Tumor")

    if not abnormalities:
        return "No Major Abnormality Detected"
    if len(abnormalities) == 1:
        return f"{abnormalities[0]} Detected"
    return "Multiple Abnormalities Detected"


def save_output(img_path, img, fatty, ct, morph, final):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = os.path.splitext(os.path.basename(img_path))[0]

    json_path = os.path.join(OUTPUT_DIR, f"{name}_{ts}.json")
    png_path = os.path.join(OUTPUT_DIR, f"{name}_{ts}.png")

    data = {
        "fatty": fatty,
        "ct": ct,
        "morphology": morph,
        "final": final
    }

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

    title = f"{final}\nFatty: {fatty['confidence']}% | CT: {ct['confidence']}%"

    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()

    return json_path, png_path


def main(img_path):
    fatty_model, ct_model = load_models()
    img, arr = preprocess(img_path)

    fatty = fatty_prediction(fatty_model, arr)
    ct = ct_prediction(ct_model, arr)
    morph = morphology(img_path)

    final = final_decision(fatty, ct)

    json_path, png_path = save_output(img_path, img, fatty, ct, morph, final)

    print("\n===== FINAL RESULT =====")
    print("Fatty:", fatty)
    print("CT:", ct)
    print("Morphology:", morph)
    print("Final:", final)
    print("Saved:", json_path, png_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/combined/combined_predict.py <image_path>")
    else:
        main(sys.argv[1])