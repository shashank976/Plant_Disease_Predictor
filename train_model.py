import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

base_dir = "plant_village/PlantVillage"
classes = [c for c in os.listdir(base_dir) if not c.startswith('.')]
print("Number of classes:", len(classes))
print("Sample classes:", classes[:10])

# Image size (small so training is fast)
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Split into training and validation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    base_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    base_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

base_model=MobileNetV2(
    input_shape=(128,128,3), 
    include_top=False, 
    weights="imagenet"
)
base_model.trainable = False 

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(len(classes), activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

EPOCHS = 5

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

model.save("plant_disease_model.h5")
print("Model saved as plant_disease_model.h5")
