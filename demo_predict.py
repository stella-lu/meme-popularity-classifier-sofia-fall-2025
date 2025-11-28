import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

IMG_SIZE = (224, 224)
DATA_DIR = "image_data"  # same as training
# This is the 2nd one with more epochs and trained on larger images
MODEL_PATH = "results/10_epochs_larger_image/meme_popularity_model_2.h5"

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Rebuild a small generator just to recover class_indices
datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)
dummy_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=1,
    class_mode="categorical",
    subset="training"
)

class_indices = dummy_gen.class_indices  # e.g. {'high': 0, 'low': 1, 'medium': 2}
idx_to_class = {v: k for k, v in class_indices.items()}
print("Class mapping:", idx_to_class)

def predict_meme(img_path: str):
    img_path = Path(img_path)

    # load and preprocess image
    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x = x / 255.0  # same rescaling as training
    x = np.expand_dims(x, axis=0)  # batch dimension

    # run prediction
    probs = model.predict(x)[0]
    pred_idx = np.argmax(probs)
    pred_label = idx_to_class[pred_idx]

    # nice printout
    print(f"\nImage: {img_path.name}")
    for i, p in enumerate(probs):
        print(f"  P({idx_to_class[i]}): {p:.3f}")
    print(f"--> Predicted label: {pred_label.upper()}")

    # show image with title
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Predicted: {pred_label}")
    plt.show()

if __name__ == "__main__":
    # Update the file paths here
    demo_files = [
        "demo_memes/47_upvotes.png",
        "demo_memes/5500_upvotes.jpeg",
    ]

    for f in demo_files:
        if Path(f).exists():
            predict_meme(f)
        else:
            print(f"File not found: {f}")
