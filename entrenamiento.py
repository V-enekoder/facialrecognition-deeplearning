import os

import matplotlib.pyplot as plt
import numpy as np  # <--- Corregido: Importación necesaria para guardar la historia
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- CONFIGURACIÓN PARA LINUX / WSL ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "dataset-procesado")
model_save_path = os.path.join(BASE_DIR, "sujetos.keras")
history_save_path = os.path.join(
    BASE_DIR, "history.npy"
)  # <--- Nuevo: Ruta para la historia

IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 40

# Aumentación de datos: Ayuda a que la IA aprenda a ignorar el fondo del gym
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.3,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.4, 1.6],
    fill_mode="nearest",
)

train_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    class_mode="categorical",
    subset="training",
    shuffle=True,
)

val_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    class_mode="categorical",
    subset="validation",
)

num_classes = len(train_gen.class_indices)

# --- ARQUITECTURA DEL MODELO ---
model = Sequential(
    [
        Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)
        ),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(256, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.6),  # Evita que el modelo memorice (Overfitting)
        Dense(num_classes, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()

# Detener el entrenamiento si el error deja de bajar (evita perder tiempo)
early_stop = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)

# --- INICIO DEL ENTRENAMIENTO ---
history = model.fit(
    train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[early_stop]
)

# --- GUARDAR RESULTADOS ---
model.save(model_save_path)
print(f"\n✅ Modelo guardado en: {model_save_path}")

# Guardar la historia de entrenamiento para analisis.py
np.save(history_save_path, history.history)
print(f"✅ Historia guardada en: {history_save_path}")

# --- GRÁFICAS INMEDIATAS ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="entrenamiento", color="blue")
plt.plot(history.history["val_accuracy"], label="validacion", color="orange")
plt.title("Precisión (Accuracy)")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="entrenamiento", color="blue")
plt.plot(history.history["val_loss"], label="validacion", color="orange")
plt.title("Error (Loss)")
plt.legend()

plt.tight_layout()
plt.show()
