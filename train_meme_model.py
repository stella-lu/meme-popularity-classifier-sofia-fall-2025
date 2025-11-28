import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = "image_data"

datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=42
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=42
)

base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dense(train_gen.num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

model.save("meme_popularity_model_2.h5")

plt.figure()
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training vs validation accuracy")
plt.savefig("accuracy_curve_2.png", dpi=150)
plt.close()

val_gen.reset()
y_true = val_gen.classes
class_indices = val_gen.class_indices
idx_to_class = {v: k for k, v in class_indices.items()}

y_pred_probs = model.predict(val_gen)
y_pred = np.argmax(y_pred_probs, axis=1)

print("Classification report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=[idx_to_class[i] for i in range(len(idx_to_class))]
))

cm = confusion_matrix(y_true, y_pred)

plt.figure()
plt.imshow(cm, cmap="Blues")
plt.title("Confusion matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.colorbar()
plt.xticks(
    ticks=range(len(idx_to_class)),
    labels=[idx_to_class[i] for i in range(len(idx_to_class))],
    rotation=45
)
plt.yticks(
    ticks=range(len(idx_to_class)),
    labels=[idx_to_class[i] for i in range(len(idx_to_class))]
)
plt.tight_layout()
plt.savefig("confusion_matrix_2.png", dpi=150)
plt.close()
