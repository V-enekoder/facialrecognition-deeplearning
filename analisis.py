import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- CONFIGURACIÓN ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "sujetos.keras")
dataset_path = os.path.join(BASE_DIR, "dataset-procesado")
output_dir = os.path.join(BASE_DIR, "metricas")

# Crear carpeta de métricas si no existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Carpeta creada: {output_dir}")

# 1. Cargar modelo y datos
modelo = load_model(model_path)
datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

val_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
)

clases = list(val_gen.class_indices.keys())
Y_pred = modelo.predict(val_gen)
y_pred = np.argmax(Y_pred, axis=1)
y_true = val_gen.classes

# --- GENERAR IMÁGENES POR SEPARADO ---

# 1. MATRIZ DE CONFUSIÓN
"""
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", xticklabels=clases, yticklabels=clases
)
plt.title("Matriz de Confusión")
plt.ylabel("Real")
plt.xlabel("Predicción")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "1_matriz_confusion.png"))
plt.close()
"""

cm = confusion_matrix(y_true, y_pred)

# Obtenemos índices específicos
indices_mariana = [i for i, c in enumerate(clases) if c.startswith("mariana")]
indices_victor = [i for i, c in enumerate(clases) if c.startswith("victor")]


def guardar_matriz_pura(indices, nombre_sujeto, filename):
    # Recortamos la matriz tanto en filas como en columnas (Sub-matriz 5x5)
    # cm[indices, :][:, indices] selecciona el bloque donde la persona es real Y predicha
    sub_cm = cm[np.ix_(indices, indices)]

    # Limpiamos los nombres para el eje (ej: 'mariana_alegre' -> 'Alegre')
    etiquetas_cortas = [clases[i].split("_")[1].capitalize() for i in indices]

    plt.figure(figsize=(6, 5))  # Tamaño cuadrado, ideal para una columna
    sns.set_context("paper", font_scale=1.2)

    sns.heatmap(
        sub_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=etiquetas_cortas,
        yticklabels=etiquetas_cortas,
        cbar=True,
        square=True,  # Fuerza a que las celdas sean cuadrados perfectos
    )

    plt.ylabel(f"Real ({nombre_sujeto})", fontweight="bold")
    plt.xlabel(f"Predicho ({nombre_sujeto})", fontweight="bold")
    plt.xticks(rotation=0)  # Al ser pocas clases, no hace falta rotar
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()


# Generar las dos matrices 5x5
guardar_matriz_pura(indices_mariana, "Mariana", "matrix_mariana_5x5.png")
guardar_matriz_pura(indices_victor, "Víctor", "matrix_victor_5x5.png")

print(f"[OK] Matrices 5x5 guardadas en '{output_dir}/'")

# 3. HISTOGRAMA DE CONFIANZA
plt.figure(figsize=(10, 6))
confianzas = np.max(Y_pred, axis=1)
plt.hist(confianzas, bins=20, color="orange", edgecolor="black")
plt.title("Nivel de Confianza de las Predicciones")
plt.xlabel("Probabilidad")
plt.ylabel("Cantidad de Imágenes")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "3_histograma_confianza.png"))
plt.close()

# 4. REPORTE DE MÉRICAS (HEATMAP)
plt.figure(figsize=(12, 8))
reporte = classification_report(y_true, y_pred, target_names=clases, output_dict=True)
reporte_df = pd.DataFrame(reporte).iloc[:-1, :].T
sns.heatmap(reporte_df, annot=True, cmap="RdYlGn", fmt=".2f")
plt.title("Métricas por Clase (Precision, Recall, F1)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "4_reporte_metricas.png"))
plt.close()


def limpiar_clase(nombre):
    partes = nombre.split("_")
    return f"{partes[0][0].upper()}: {partes[1].capitalize()}"  # Ej: M: Alegre


clases_limpias = [limpiar_clase(c) for c in clases]

# --- 4. REPORTE DE MÉRICAS (HEATMAP SIMPLIFICADO) ---
plt.figure(figsize=(7, 5))  # Proporción compacta
sns.set_context("paper", font_scale=1.2)

reporte = classification_report(
    y_true, y_pred, target_names=clases_limpias, output_dict=True
)
# Solo nos interesan las clases individuales, quitamos 'accuracy', 'macro avg', etc.
reporte_df = pd.DataFrame(reporte).iloc[:-1, : len(clases)].T

sns.heatmap(
    reporte_df,
    annot=True,
    cmap="RdYlGn",
    fmt=".2f",
    cbar=False,  # Quitamos la barra de color para ganar espacio horizontal
)

plt.title("Métricas de Clasificación por Sujeto y Emoción", fontweight="bold")
plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "4_reporte_metricas_paper.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# --- 2. DISTRIBUCIÓN (SESGO - MÁS LIMPIO) ---
plt.figure(figsize=(7, 4))
sns.set_context("paper", font_scale=1.1)

conteo = [np.sum(y_true == i) for i in range(len(clases))]

# Usamos un color distinto para cada persona para que el sesgo se vea a simple vista
colores = ["#4A90E2" if "M:" in c else "#50C878" for c in clases_limpias]

sns.barplot(x=conteo, y=clases_limpias, palette=colores)

plt.title("Distribución del Dataset de Validación", fontweight="bold")
plt.xlabel("Número de Imágenes")
plt.ylabel("")  # Quitamos el label 'Clases' porque es obvio
plt.grid(axis="x", linestyle="--", alpha=0.7)  # Una rejilla ligera ayuda a leer
plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "2_sesgo_dataset_paper.png"), dpi=300, bbox_inches="tight"
)
plt.close()

print(f"[OK] Gráficas de Reporte y Distribución optimizadas en '{output_dir}/'")


history_path = os.path.join(BASE_DIR, "history.npy")
if os.path.exists(history_path):
    h = np.load(history_path, allow_pickle=True).item()
    plt.figure(figsize=(12, 5))
    # ... (código de plt.plot usando 'h' en lugar de 'history.history')
    plt.subplot(1, 2, 1)
    plt.plot(h["accuracy"], label="Entrenamiento")
    plt.plot(h["val_accuracy"], label="Validación")
    plt.title("Precisión (Accuracy)")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(h["loss"], label="Entrenamiento")
    plt.plot(h["val_loss"], label="Validación")
    plt.title("Error (Loss)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "5_curvas_aprendizaje.png"))
    plt.close()

print(f"\n[OK] Se han guardado 5 imágenes en la carpeta '{output_dir}/'")
