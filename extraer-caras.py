import os
import cv2

CARPETA_SALIDA = "dataset-procesado"
LIMITE_IMAGENES = 210  # Límite solicitado

VIDEOS = {
    "victor_alegre": ["videos/victor-feliz.mp4", "videos/victor-feliz2.mp4", "videos/victor-feliz3.mp4"],
    "victor_triste": ["videos/victor-triste.mp4", "videos/victor-triste2.mp4", "videos/victor-triste3.mp4"],
    "victor_molesto": ["videos/victor-molesto.mp4", "videos/victor-molesto2.mp4", "videos/victor-molesto3.mp4"],
    "victor_neutral": ["videos/victor-neutral.mp4", "videos/victor-neutral2.mp4", "videos/victor-neutral3.mp4"],
    "victor_sorprendido": [
        "videos/victor-sorprendido.mp4",
        "videos/victor-sorprendido2.mp4",
        "videos/victor-sorprendido3.mp4",
    ],
    "mariana_alegre": [
        "videos/mariana-feliz.mp4",
        "videos/mariana-feliz2.mp4",
        "videos/mariana-feliz3.mp4",
    ],
    "mariana_molesta": [
        "videos/mariana-molesta.mp4",
        "videos/mariana-molesta2.mp4",
        "videos/mariana-molesta3.mp4",
    ],
    "mariana_neutral": [
        "videos/mariana-neutral.mp4",
        "videos/mariana-neutral2.mp4",
        "videos/mariana-neutral3.mp4",
    ],
    "mariana_sorprendida": [
        "videos/mariana-sorprendida.mp4",
        "videos/mariana-sorprendida2.mp4",
        "videos/mariana-sorprendida3.mp4",
    ],
    "mariana_triste": [
        "videos/mariana-triste.mp4",
        "videos/mariana-triste2.mp4",
        "videos/mariana-triste3.mp4",
    ],
}

FRECUENCIA_CUADROS = 10
MARGIN = 0

def crear_directorios():
    if not os.path.exists(CARPETA_SALIDA):
        os.makedirs(CARPETA_SALIDA)
        print(f"📁 Carpeta '{CARPETA_SALIDA}' creada.")

def extraer_frames():
    if not os.path.exists(CARPETA_SALIDA):
        os.makedirs(CARPETA_SALIDA)

    for emocion, lista_videos in VIDEOS.items():
        print(f"\n🎬 Procesando clase: '{emocion}'...")

        ruta_clase = os.path.join(CARPETA_SALIDA, emocion)
        if not os.path.exists(ruta_clase):
            os.makedirs(ruta_clase)

        total_guardadas_clase = 0

        for ruta_video in lista_videos:
            # Si ya alcanzamos las 210 para esta emoción, dejamos de procesar sus videos
            if total_guardadas_clase >= LIMITE_IMAGENES:
                break

            if not os.path.exists(ruta_video):
                print(f"  ⚠️  No encontré el video: {ruta_video} (Saltando...)")
                continue

            print(f"  📽️  Extrayendo de: {ruta_video}...")

            cap = cv2.VideoCapture(ruta_video)
            count = 0
            guardadas_este_video = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break 

                # Verificación de límite dentro del bucle del video
                if total_guardadas_clase >= LIMITE_IMAGENES:
                    break

                if count % FRECUENCIA_CUADROS == 0:
                    nombre_archivo = f"{emocion}_{total_guardadas_clase}.jpg"
                    ruta_final = os.path.join(ruta_clase, nombre_archivo)

                    cv2.imwrite(ruta_final, frame)
                    guardadas_este_video += 1
                    total_guardadas_clase += 1

                count += 1

            cap.release()
            print(f"    ✅ {guardadas_este_video} imágenes extraídas de este archivo.")

        print(f"✨ Total final para /{emocion}: {total_guardadas_clase} imágenes")

    print(f"\n🚀 ¡Proceso terminado! Máximo {LIMITE_IMAGENES} por carpeta en '{CARPETA_SALIDA}'")

if __name__ == "__main__":
    extraer_frames()