from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import base64
import io
import os
from PIL import Image
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import json
app = Flask(__name__, static_folder="/home/ec2-user/pintalo_backend/processed/")
#app = Flask(__name__, static_folder="/Users/inigoliberal/Desktop/PINTALO/pintalo_backend/processed/")

CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

current_filename = ""


def hex_a_rgb(hex_colores):
    return np.array([tuple(int(color[i:i+2], 16) for i in (1, 3, 5)) for color in hex_colores], dtype=np.uint8)


@app.route("/convert", methods=["POST",'GET'])
def convert_to_black_and_white():
    #Consigo lo que viene del formdata
    colores = request.form.get("colores")

    if colores:
        try:
            colores = json.loads(colores)  # Convertir a lista de Python
        except json.JSONDecodeError:
            colores = []  # O manejar el error de conversión
    print('llego')
    print(colores)
    print(len(colores))


    #Si no hay colores, hago lo de siempre

    if len(colores)==0:
        print('entro en no')
        #Decodificar imagen y prepararla
        image=request.form.get('file')

        image_base64 = image.split(",")[1]  # Elimina "data:image/png;base64,"

        image_decoded = base64.b64decode(image_base64)

        image_ready = Image.open(io.BytesIO(image_decoded))

        image_np = np.array(image_ready)  # Convertir a NumPy (RGB)

        '''
        'ESTILO DIBUJO'

        # Convertir a escala de grises
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

        # Aplicar desenfoque para suavizar
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)

        # Crear el efecto de boceto
        bordes_invertidos = cv2.divide(gray, blurred, scale=256)
        '''
        #Convertir a formato BGR para OpenCV
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Aplanar la imagen para usar en K-Means. Convertirlo a matriz
        Z = image_bgr.reshape((-1, 3))  # Convertir a una matriz 2D de píxeles
        Z = np.float32(Z)

        # Definir criterios de K-Means y aplicar el algoritmo
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(Z, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) #EL 5 ES EL NUMERO DE SEGMENTOS(COLORES), los devuelve en centers

        #print(centers)#Estos colores son los principales, los segmentados

        #Convertir los centros a valores enteros y reconstruir la imagen segmentada
        centers = np.uint8(centers)
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(image_bgr.shape)

        # 6️⃣ Convertir a escala de grises y encontrar bordes. Thresh es una imagen binaria para luego buscarle los contornos.
        gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 7️⃣ Encontrar contornos. Cada contour es un punto x,y que me dvuelve la posicion del contorno
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #print(contours)

        image_bn= cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

        bordes = cv2.Canny(image_bn, 50, 100) #100 y 500 son los limites para encontrar el borde, variandolos consigo refinar mas o menos

        # Invertir la imagen (opcional)
        bordes_invertidos = cv2.bitwise_not(bordes)
        i=0
        j=0
        # 8️⃣ Dibujar números en cada segmento
        for i, contour in enumerate(contours): #Enumero cada contorno y lo recorro
            M = cv2.moments(contour) #Calculo el centro del contorno
            #print(M)
            if M["m00"] != 0:  # Evitar divisiones por cero, quiere decir que el segmento es muy pequenio
                i+=1
                print('pinto',i)
                cx = int(M["m10"] / M["m00"])  # Coordenada X del centro del segmento
                cy = int(M["m01"] / M["m00"])  # Coordenada Y del centro del segmento
                cv2.putText(bordes_invertidos, str(i+1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            else:
                j+=1
                print('no pinto',j)
        

        # Mostrar la imagen original y la segmentada
        cv2.imwrite("/home/ec2-user/pintalo_backend/processed/processed_image.png", bordes_invertidos)
        #cv2.imwrite("/Users/inigoliberal/Desktop/PINTALO/pintalo_backend/processed/processed_image.png", bordes_invertidos)

        # Guardar la imagen en blanco y negro

        with open("/home/ec2-user/pintalo_backend/processed/processed_image.png", "rb") as image_file:
            # Codificar la imagen en Base64
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        # Enviar la imagen en JSON
        return jsonify({"image_base64": encoded_string})
    
    #Si hay colores
    else:
        print('colores;',colores)
        #Decodificar imagen y prepararla
        colores_rgb=hex_a_rgb(colores)

        print(colores_rgb)

        image=request.form.get('file')

        image_base64 = image.split(",")[1]  # Elimina "data:image/png;base64,"

        image_decoded = base64.b64decode(image_base64)

        image_ready = Image.open(io.BytesIO(image_decoded))

        image_np = np.array(image_ready)  # Convertir a NumPy (RGB)

        #Convertir a formato BGR para OpenCV
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Aplanar la imagen para usar en K-Means. Convertirlo a matriz
        pixeles = image_bgr.reshape((-1, 3))

        kmeans = KMeans(n_clusters=len(colores_rgb), init=colores_rgb.astype(np.float32), n_init=1, max_iter=10)
        labels = kmeans.fit_predict(pixeles)

        # 📌 Asignar cada píxel a su color más cercano (usando los centroides)
        bordes_invertidos = colores_rgb[labels].reshape(image_bgr.shape)

        # Mostrar la imagen original y la segmentada
        cv2.imwrite("/home/ec2-user/pintalo_backend/processed/processed_image.png", bordes_invertidos)
        #cv2.imwrite("/Users/inigoliberal/Desktop/PINTALO/pintalo_backend/processed/processed_image.png", bordes_invertidos)

        # Guardar la imagen en blanco y negro

        with open("/home/ec2-user/pintalo_backend/processed/processed_image.png", "rb") as image_file:
            # Codificar la imagen en Base64
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        # Enviar la imagen en JSON
        return jsonify({"image_base64": encoded_string})

    #return jsonify({"success": True, "image_url": f"/home/ec2-user/pintalo_backend/processed/processed_image.png"})


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/processed/<filename>")
def processed_file(filename):
    return send_from_directory(app.config["PROCESSED_FOLDER"], filename)


@app.route('/download/')
def serve_image():
    # O, si tienes la imagen en memoria (en un objeto BytesIO por ejemplo):
    processed_image = open("/home/ec2-user/pintalo_backend/processed/processed_image.png", "rb").read()  # Leer archivo binario
    #processed_image = open("/Users/inigoliberal/Desktop/PINTALO/pintalo_backend/processed/processed_image.png", "rb").read()

    # Crear un objeto BytesIO con la imagen procesada en memoria
    image_io = io.BytesIO(processed_image)

    # Devolver el archivo como una descarga
    return send_file(image_io, mimetype='image/png', as_attachment=True, download_name='imagen_procesada.png')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
