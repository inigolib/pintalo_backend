from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import base64
import io
import os
from PIL import Image

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
    image=request.form.get('file')
    image_base64 = image.split(",")[1]  # Elimina "data:image/png;base64,"

    image_decoded = base64.b64decode(image_base64)

    image_para_bn = Image.open(io.BytesIO(image_decoded))
    umbral = 128  # Valores menores a 128 serán negras, mayores serán blancas
    image_bn = image_para_bn.convert("L").point(lambda x: 255 if x > umbral else 0, mode="1")

    # Guardar la imagen en blanco y negro
    image_bn.save("/Users/inigoliberal/Desktop/PROYECTO PINTALO/processed/processed_image.png")

    print('llego')

    return jsonify({"success": True, "image_url": f"/Users/inigoliberal/Desktop/PROYECTO PINTALO/processed/processed_image.png"})


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/processed/<filename>")
def processed_file(filename):
    return send_from_directory(app.config["PROCESSED_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True)