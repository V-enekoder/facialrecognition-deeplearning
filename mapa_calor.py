import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Ocultar advertencias molestas
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings

warnings.filterwarnings("ignore")

# --- 1. CONFIGURACIÓN DE RUTAS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "sujetos.keras")
ruta_imagen = os.path.join(BASE_DIR, "imagenes-prueba", "mariana-neutral.jpeg")

# --- 2. VALIDACIONES ---
if not os.path.exists(model_path):
    print(f"❌ Error: No se encuentra el modelo en {model_path}")
    exit()
if not os.path.exists(ruta_imagen):
    print(f"❌ Error: No se encuentra la imagen en {ruta_imagen}")
    exit()

print("Cargando modelo...")
modelo = tf.keras.models.load_model(model_path)

# --- 3. ENCONTRAR EL ÍNDICE DE LA ÚLTIMA CAPA CONVOLUCIONAL ---
indice_capa_conv = -1
for i, layer in enumerate(modelo.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        indice_capa_conv = i

if indice_capa_conv == -1:
    raise ValueError("No se encontró ninguna capa Conv2D en el modelo.")

print(
    f"Última capa convolucional en el índice: {indice_capa_conv} ({modelo.layers[indice_capa_conv].name})"
)

# --- 4. PREPARAR LA IMAGEN ---
img_tensor = image.load_img(ruta_imagen, target_size=(128, 128), color_mode="grayscale")
img_array = image.img_to_array(img_tensor) / 255.0
img_array = np.expand_dims(img_array, axis=0)

img_original = cv2.imread(ruta_imagen)
img_original = cv2.resize(img_original, (128, 128))

# --- 5. PASO MANUAL Y CÁLCULO DE GRADIENTES (MÉTODO A PRUEBA DE FALLOS) ---
# Convertimos el array a un Tensor de TensorFlow explícitamente
x = tf.convert_to_tensor(img_array, dtype=tf.float32)

with tf.GradientTape() as tape:
    # 5.1 Pasar la imagen capa por capa hasta la última convolucional
    for i in range(indice_capa_conv + 1):
        x = modelo.layers[i](x, training=False)

    # 5.2 ¡AQUÍ ESTÁ NUESTRO MAPA DE CARACTERÍSTICAS! Lo guardamos y lo vigilamos
    capas_conv_outputs = x
    tape.watch(capas_conv_outputs)

    # 5.3 Pasar por el resto de las capas (Flatten, Dense, etc.)
    for i in range(indice_capa_conv + 1, len(modelo.layers)):
        x = modelo.layers[i](x, training=False)

    predicciones = x

    # 5.4 Obtener el error (loss) de la clase ganadora
    indice_clase_ganadora = tf.argmax(predicciones[0])
    loss = predicciones[:, indice_clase_ganadora]

# --- 6. CONSTRUIR EL MAPA DE CALOR ---
# Calcular gradientes
grads = tape.gradient(loss, capas_conv_outputs)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

capas_conv_outputs = capas_conv_outputs[0]
heatmap = capas_conv_outputs @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)

# Normalizar (1e-10 evita división por 0)
heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
heatmap = heatmap.numpy()

# --- 7. SUPERPONER EL MAPA EN LA FOTO ---
heatmap_redimensionado = cv2.resize(
    heatmap, (img_original.shape[1], img_original.shape[0])
)
heatmap_color = np.uint8(255 * heatmap_redimensionado)
heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)

imagen_superpuesta = cv2.addWeighted(img_original, 0.6, heatmap_color, 0.4, 0)

# --- 8. MOSTRAR RESULTADOS ---
plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
plt.title("Foto Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(heatmap_redimensionado, cmap="jet")
plt.title("Zonas de Interés (Red)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(imagen_superpuesta, cv2.COLOR_BGR2RGB))
plt.title("Grad-CAM Superpuesto")
plt.axis("off")

plt.tight_layout()
plt.show()
