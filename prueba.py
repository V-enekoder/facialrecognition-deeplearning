import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- CONFIGURACIÓN ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "sujetos.keras")
dataset_path = os.path.join(BASE_DIR, "dataset-procesado")
prueba_dir = os.path.join(BASE_DIR, "imagenes-prueba")

# Cargar modelo
if not os.path.exists(model_path):
    print(f"Error: No se encuentra {model_path}")
    exit()

modelo = tf.keras.models.load_model(model_path)

# Obtener nombres de clases (para mapear el índice a texto)
datagen = ImageDataGenerator(rescale=1.0 / 255)
temp_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=1,
    class_mode="categorical",
    shuffle=False,
)
clases = list(temp_gen.class_indices.keys())

# --- PROCESAR CARPETA DE PRUEBAS ---
if not os.path.exists(prueba_dir):
    print(f"Error: No existe la carpeta {prueba_dir}")
    exit()

# Extensiones válidas
extensiones = (".jpg", ".jpeg", ".png")
archivos = [f for f in os.listdir(prueba_dir) if f.lower().endswith(extensiones)]
archivos.sort()  # Ordenar alfabéticamente

print("\n" + "=" * 85)
print(f"{'ARCHIVO (Real)':<30} | {'PREDICCIÓN':<20} | {'CONFIDENCIA':<12} | {'ESTADO'}")
print("-" * 85)

aciertos = 0

for nombre_archivo in archivos:
    ruta_completa = os.path.join(prueba_dir, nombre_archivo)

    # Cargar y procesar imagen
    img = image.load_img(ruta_completa, target_size=(128, 128), color_mode="grayscale")

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predicción
    pred = modelo.predict(img_array, verbose=0)
    indice = np.argmax(pred)
    clase_predicha = clases[indice]
    confianza = pred[0][indice] * 100

    # Normalizamos nombres para comparar (quitar guiones, extensiones, etc)
    nombre_clean = nombre_archivo.lower().replace("-", "_").split(".")[0]
    # Caso especial: 'feliz' en archivo suele ser 'alegre' en tu dataset
    nombre_clean = nombre_clean.replace("feliz", "alegre")

    match = (
        "✅"
        if (
            clase_predicha.lower() in nombre_clean
            or nombre_clean in clase_predicha.lower()
        )
        else "❌"
    )
    if match == "✅":
        aciertos += 1

    print(
        f"{nombre_archivo:<30} | {clase_predicha:<20} | {confianza:>10.2f}% | {match}"
    )

print("-" * 85)
total = len(archivos)
print(
    f"RESUMEN: {aciertos}/{total} aciertos ({(aciertos / total) * 100:.2f}% de precisión en pruebas)"
)
print("=" * 85 + "\n")
