from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from tensorflow import keras
from keras.models import load_model
from keras_facenet import FaceNet
labels=['Agus','Ale', 'Noni', 'Sebas']
app = Flask(__name__)
CORS(app) 

model=None

embedder = FaceNet()
# Función para generar el embedding usando FaceNet
def calculate_face_net(images):
    images=images.astype('float32')
    images = np.expand_dims(images, axis=0)
    embeddings = embedder.embeddings(images)
    #print(embeddings[0])
    return embeddings[0] 

def preprocess_images(img):
    """
    Preprocesa una lista de imágenes: las convierte a RGB, redimensiona y normaliza
    """
    img=cv2.cvtColor(cv2.resize(img, (160, 160)), cv2.COLOR_BGR2RGB)
    img=np.array(img)

    #print(np.array(processed_images))
    return img  # Devuelve un lote de imágenes


def normalize_matrix(matrix):
    normalized_matrix = np.zeros_like(matrix)  # Crear matriz del mismo tamaño
    for i in range(matrix.shape[0]):  # Iterar por cada fila (embedding)
        X_min = np.min(matrix[i])  # Mínimo de la fila
        X_max = np.max(matrix[i])  # Máximo de la fila
        if X_max != X_min:  # Evitar división por cero
            normalized_matrix[i] = (matrix[i] - X_min) / (X_max - X_min)
        else:
            normalized_matrix[i] = matrix[i]  # Si min == max, no se normaliza
    return normalized_matrix



def get_model():
    return load_model("./modelo14.h5") if model is None else model
@app.route('/upload', methods=['POST'])
def upload():
    datos = request.get_json()
    imagen_base64 = datos['imagen']
    imagen_bytes = base64.b64decode(imagen_base64)
    np_arr = np.frombuffer(imagen_bytes, np.uint8)
    imagen = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    model=get_model()
    imagenCopia=imagen.copy()
    # Procesar la imagen (escala de grises)
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    classificador=cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
    porcentajex=0.15
    porcentajey=0.05
    rostroEscalaDeGrises=imagen_gris
    caras=classificador.detectMultiScale(
                        rostroEscalaDeGrises,
                        scaleFactor=1.1,
                        minNeighbors=8         
            )
    for cara in caras:
                  x,y,w,h=cara
                  cv2.rectangle(imagen,(x,y),(x+w,y+h),(255,0,0),2)
                  reducciony=int(h*porcentajey)
                  reduccionx=int(w*porcentajex)
                  rostro=imagenCopia[y+reducciony:y+h-reducciony,x+reduccionx:x+w-reduccionx]
                  if rostro.size>0:
                    rostroReducido= cv2.resize(rostro, (200, 200))
                    print(np.array(rostroReducido).shape)#200x200x3
                    rostroReducidoProcessed=preprocess_images(rostroReducido)
                    print(rostroReducidoProcessed.shape)
                    embedding=calculate_face_net(rostroReducidoProcessed)
                    print("Embedding",embedding.shape)
                    embedding = np.expand_dims(embedding, axis=0)  # Ahora tendrá forma (1, 512)
                    embedding = normalize_matrix(embedding)

                    print("Embedding2",embedding.shape)

                    porcentajes=model.predict(embedding)
                    persona=labels[np.argmax(porcentajes)] if np.max(porcentajes)>0.50 else "Desconocido" 
                    cv2.putText(imagen,f"{persona}"
                                ,(x,y-50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0),2)
                    cv2.putText(imagen,f"{np.max(porcentajes):.2f}"
                                ,(x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0),2)
    _, buffer = cv2.imencode('.jpg', imagen)
    imagen_gris_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'imagen_procesada': imagen_gris_base64,'label':persona}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
