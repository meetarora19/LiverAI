import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, f1_score
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image_dataset_from_directory

BASE_DIR = r"D:\LiverAI_Project"

TRAIN_DIR = os.path.join(BASE_DIR, "data", "ct_liver", "processed", "splits", "train")
VAL_DIR = os.path.join(BASE_DIR, "data", "ct_liver", "processed", "splits", "val")
TEST_DIR = os.path.join(BASE_DIR, "data", "ct_liver", "processed", "splits", "test")

MODEL_DIR = os.path.join(BASE_DIR, "models", "ct_model", "saved_models")
PLOTS_DIR = os.path.join(BASE_DIR, "data", "ct_liver", "outputs", "visualizations")
METRICS_DIR = os.path.join(BASE_DIR, "data", "ct_liver", "outputs", "metrics")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
EPOCHS = 15

train_ds = image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
)

val_ds = image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
)

test_ds = image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=False
)

class_names = train_ds.class_names
print("Classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(1)
val_ds = val_ds.prefetch(1)
test_ds = test_ds.prefetch(1)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.15),
])

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        patience=2,
        factor=0.5
    ),
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "ct_classifier_checkpoint.keras"),
        save_best_only=True,
        monitor="val_loss"
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("Epochs actually run:", len(history.history["loss"]))

plt.figure(figsize=(6, 4))
plt.plot(history.history["accuracy"], label="train_accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.title("CT Slice Classifier Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "ct_classifier_accuracy.png"))
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("CT Slice Classifier Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "ct_classifier_loss.png"))
plt.show()

val_loss, val_acc = model.evaluate(val_ds, verbose=0)
test_loss, test_acc = model.evaluate(test_ds, verbose=0)

print("Validation Accuracy:", val_acc)
print("Test Accuracy:", test_acc)

y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    pred_labels = np.argmax(preds, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(pred_labels)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
cm = confusion_matrix(y_true, y_pred)
balanced_acc = balanced_accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro")

print(report)
print("Confusion Matrix:\n", cm)
print("Balanced Accuracy:", balanced_acc)
print("Macro F1 Score:", macro_f1)

with open(os.path.join(METRICS_DIR, "ct_classifier_report.txt"), "w") as f:
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(np.array2string(cm))
    f.write(f"\nBalanced Accuracy: {balanced_acc:.6f}")
    f.write(f"\nMacro F1 Score: {macro_f1:.6f}")

model_path = os.path.join(MODEL_DIR, "ct_classifier.keras")
model.save(model_path)
print("Model saved to:", model_path)

results_df = pd.DataFrame({
    "Metric": ["Validation Accuracy", "Test Accuracy", "Balanced Accuracy", "Macro F1"],
    "Value": [val_acc, test_acc, balanced_acc, macro_f1]
})
results_df.to_csv(os.path.join(METRICS_DIR, "ct_classifier_results.csv"), index=False)
print("Metrics saved successfully.")