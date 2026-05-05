import os
import numpy as np
import tensorflow as tf

from utils.metrics_utils import (
    compute_metrics,
    save_metrics,
    plot_confusion_matrix,
    save_classification_report
)

# ===== CONFIG =====
MODEL_PATH = "models/ct_model/saved_models/model.keras"
TEST_DIR = "data/ct_liver/processed/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ===== LOAD MODEL =====
model = tf.keras.models.load_model(MODEL_PATH)

# ===== LOAD DATA =====
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_dataset.class_names
print("Classes:", class_names)

# ===== PREDICTION LOOP =====
y_true = []
y_pred = []

for images, labels in test_dataset:
    preds = model.predict(images)

    preds = np.argmax(preds, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(preds)

# ===== EVALUATION =====
output_dir = "data/ct_liver/outputs/metrics"
os.makedirs(output_dir, exist_ok=True)

metrics = compute_metrics(y_true, y_pred)
save_metrics(metrics, os.path.join(output_dir, "ct_metrics.json"))

plot_confusion_matrix(
    y_true, y_pred,
    class_names,
    os.path.join(output_dir, "ct_confusion_matrix.png")
)

save_classification_report(
    y_true, y_pred,
    os.path.join(output_dir, "ct_classification_report.json")
)

print("\n===== CT MODEL METRICS =====")
print(metrics)