from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from keras.models import load_model
from keras_facenet import FaceNet
import os

labels = ['Agus', 'Ale', 'Noni', 'Sebas']

app = Flask(__name__)
CORS(app, origins=["http://localhost:8100"])  # Permitir solo el origen específico

# Cargar el modelo una sola vez
model = load_model("./modelo14.h5")

# Inicialización de FaceNet
embedder = FaceNet()

# Función para generar el embedding usando FaceNet
def calculate_face_net(images):
    images = images.astype('float32')
    images = np.expand_dims(images, axis=0)
    embeddings = embedder.embeddings(images)
    return embeddings[0]

# Función para preprocesar las imágenes
def preprocess_images(img):
    """
    Preprocesa una lista de imágenes: las convierte a RGB, redimensiona y normaliza
    """
    img = cv2.cvtColor(cv2.resize(img, (160, 160)), cv2.COLOR_BGR2RGB)
    return np.array(img)

# Función para normalizar la matriz de embeddings
def normalize_matrix(matrix):
    normalized_matrix = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        X_min = np.min(matrix[i])
        X_max = np.max(matrix[i])
        if X_max != X_min:
            normalized_matrix[i] = (matrix[i] - X_min) / (X_max - X_min)
        else:
            normalized_matrix[i] = matrix[i]
    return normalized_matrix

@app.route('/upload', methods=['POST'])
def upload():
    datos = request.get_json()
    imagen_base64 = datos['imagen']
    imagen_bytes = base64.b64decode(imagen_base64)
    np_arr = np.frombuffer(imagen_bytes, np.uint8)
    imagen = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Redimensionar imagen para optimización
    imagen = cv2.resize(imagen, (640, 480))  # Redimensiona la imagen a un tamaño más manejable

    imagenCopia = imagen.copy()
    
    # Procesar la imagen (escala de grises)
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    classificador = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
    
    # Umbral mínimo para el tamaño de los rostros a procesar
    porcentajex = 0.15
    porcentajey = 0.05

    rostroEscalaDeGrises = imagen_gris
    caras = classificador.detectMultiScale(
        rostroEscalaDeGrises,
        scaleFactor=1.1,
        minNeighbors=8
    )

    persona = "Desconocido"  # Valor por defecto si no se detectan caras

    for cara in caras:
        x, y, w, h = cara
        if w > 50 and h > 50:  # Solo procesar rostros con tamaño mínimo
            cv2.rectangle(imagen, (x, y), (x + w, y + h), (255, 0, 0), 2)
            reducciony = int(h * porcentajey)
            reduccionx = int(w * porcentajex)
            rostro = imagenCopia[y + reducciony:y + h - reducciony, x + reduccionx:x + w - reduccionx]
            if rostro.size > 0:
                rostroReducido = cv2.resize(rostro, (200, 200))
                rostroReducidoProcessed = preprocess_images(rostroReducido)
                embedding = calculate_face_net(rostroReducidoProcessed)
                embedding = np.expand_dims(embedding, axis=0)
                embedding = normalize_matrix(embedding)

                # Predicción de la persona
                porcentajes = model.predict(embedding)
                persona = labels[np.argmax(porcentajes)] if np.max(porcentajes) > 0.50 else "Desconocido"

                # Añadir etiquetas en la imagen
                cv2.putText(imagen, f"{persona}", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                cv2.putText(imagen, f"{np.max(porcentajes):.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    # Codificar la imagen procesada en base64
    _, buffer = cv2.imencode('.jpg', imagen)
    imagen_gris_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'imagen_procesada': imagen_gris_base64, 'label': persona}), 200

if __name__ == '__main__':
    PORT = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=PORT)
