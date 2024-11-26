from generateresponse import generate_response
from flask import Flask, jsonify, make_response, request
import pandas as pd
from flask_cors import CORS
import json
import threading

app = Flask(__name__)

respuestas_separadas = []

CORS(app)
@app.route("/")
def hello():
    return "I am alive!"

@app.route('/autocomplete', methods=['POST'])
def autocomplete():
    texto = json.loads(request.get_data().decode("utf-8"))["texto"]
    if not json.loads(request.get_data().decode("utf-8"))["texto"] or not texto:
        texto = "HOLA"
    print(texto)
    # texto = "hola"
    respuestas = generate_response(texto)
    print(respuestas)
    # Supongamos que 'respuestas' es tu cadena original
    respuestas_separadas = respuestas.split("\n")  # Elimina la primera línea
    print(respuestas_separadas)
    # respuestas_separadas = ["Hola, soy Matias", "Hola como estas?", "Hola buen dia."]
    return jsonify(respuestas_separadas)

threading.Thread(target=app.run, kwargs={'host':'0.0.0.0','port':3000}).start()