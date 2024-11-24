from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from tensorflow import keras
from keras.models import load_model
labels=['Ale', 'Noni', 'Sebas']
app = Flask(__name__)
CORS(app) 
model=None
def get_model():
    return load_model("./CNNrostros.h5") if model is None else model
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
                    rostroEscalaDeGrises=cv2.cvtColor(rostroReducido,cv2.COLOR_BGR2GRAY)/255.0
                    rostroEscalaDeGrises=rostroEscalaDeGrises.flatten().T.reshape(1, -1)
                    print(rostroEscalaDeGrises.shape)
                    porcentajes=model.predict(rostroEscalaDeGrises)
                    persona=labels[np.argmax(porcentajes)] if np.max(porcentajes)>0.50 else "Desconocido" 
    print(imagen_gris.shape)
    _, buffer = cv2.imencode('.jpg', imagenCopia)
    imagen_gris_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'imagen_procesada': imagen_gris_base64,'label':persona}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
