import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

MODEL_PATH = "plant_disease_transfer.keras"
IMG_SIZE = 224
CLASSMAP_SOURCE_DIR = "plant_village/PlantVillage"  # to rebuild class index order

def load_class_names():
    # Rebuild the class order the model was trained with
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    gen = datagen.flow_from_directory(
        CLASSMAP_SOURCE_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=1,
        class_mode='categorical',
        subset='training',
        shuffle=False
    )
    # invert the dict: index -> class name
    inv = {v: k for k, v in gen.class_indices.items()}
    return [inv[i] for i in range(len(inv))]

def prepare_image(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    x = tf.keras.utils.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    return x

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py /path/to/leaf.jpg")
        sys.exit(1)

    img_path = sys.argv[1]
    model = load_model(MODEL_PATH)
    class_names = load_class_names()

    x = prepare_image(img_path)
    probs = model.predict(x)[0]
    idx = np.argmax(probs)
    print(f"Prediction: {class_names[idx]} (confidence: {probs[idx]:.3f})")
