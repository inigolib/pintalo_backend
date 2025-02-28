from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import base64
import io
import os
from PIL import Image
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)



UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

current_filename = ""


@app.route("/convert", methods=["POST",'GET'])
def convert_to_black_and_white():
    #Decodificar imagen y prepararla
    image=request.form.get('file')

    image_base64 = image.split(",")[1]  # Elimina "data:image/png;base64,"

    image_decoded = base64.b64decode(image_base64)

    image_ready = Image.open(io.BytesIO(image_decoded))

    image_np = np.array(image_ready)  # Convertir a NumPy (RGB)

    # 2️⃣ Convertir a formato BGR para OpenCV
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # 3️⃣ Aplanar la imagen para usar en K-Means
    Z = image_bgr.reshape((-1, 3))  # Convertir a una matriz 2D de píxeles
    Z = np.float32(Z)

    # 4️⃣ Definir criterios de K-Means y aplicar el algoritmo
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, 10, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) #EL 6 ES EL NUMERO DE SEGMENTOS(COLORES)

    # 5️⃣ Convertir los centros a valores enteros y reconstruir la imagen segmentada
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image_bgr.shape)

    # 6️⃣ Convertir a escala de grises y encontrar bordes
    gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 7️⃣ Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 8️⃣ Dibujar números en cada segmento

    image_bn= cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

    bordes = cv2.Canny(image_bn, 100, 500)

    # Invertir la imagen (opcional)
    bordes_invertidos = cv2.bitwise_not(bordes) 


    for i, contour in enumerate(contours):
        M = cv2.moments(contour)
        if M["m00"] != 0:  # Evitar divisiones por cero
            cx = int(M["m10"] / M["m00"])  # Coordenada X del centro del segmento
            cy = int(M["m01"] / M["m00"])  # Coordenada Y del centro del segmento
            cv2.putText(bordes_invertidos, str(i+1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    

    

    # Mostrar la imagen original y la segmentada
    cv2.imwrite("/Users/inigoliberal/Desktop/PINTALO/pintalo_backend/processed/processed_image.png", bordes_invertidos)



    # Guardar la imagen en blanco y negro

    return jsonify({"success": True, "image_url": f"/Users/inigoliberal/Desktop/PINTALO/pintalo_backend/processed/processed_image.png"})


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/processed/<filename>")
def processed_file(filename):
    return send_from_directory(app.config["PROCESSED_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True)