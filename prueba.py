import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy
import tflearn
import tensorflow.compat.v1 as tf
import json
import random
import pickle
import requests
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS 
import re
stemmer = LancasterStemmer()

# nltk.download('punkt')

with open("contenido.json", encoding='utf-8') as archivo:
    datos = json.load(archivo)

# Cargar contenido de "citas.json" si el tag es "citas_medicas"
with open("citas.json", encoding='utf-8') as archivo_citas:
    datos_citas = json.load(archivo_citas)
    

try:
    with open("variables.pickle", "rb") as archivoPickle:
        palabras, tags, entrenamiento, salida = pickle.load(archivoPickle)
except:
    palabras = []
    tags = []
    auxX = []
    auxY = []

    for contenido in datos["contenido"]:
        for patrones in contenido["patrones"]:
            auxPalabra = nltk.word_tokenize(patrones)
            palabras.extend(auxPalabra)
            auxX.append(auxPalabra)
            auxY.append(contenido["tag"])

            if contenido["tag"] not in tags:
                tags.append(contenido["tag"])
    
    for contenido in datos_citas["contenido"]:
        for patrones in contenido["patrones"]:
            auxPalabra = nltk.word_tokenize(patrones)
            palabras.extend(auxPalabra)
            auxX.append(auxPalabra)
            auxY.append(contenido["tag"])

            if contenido["tag"] not in tags:
                tags.append(contenido["tag"])

    palabras = [stemmer.stem(w.lower()) for w in palabras if w != "?"]
    palabras = sorted(list(set(palabras)))
    tags = sorted(tags)

    entrenamiento = []
    salida = []

    salidaVacia = [0 for _ in range(len(tags))]

    for x, documento in enumerate(auxX):
        cubeta = []
        auxPalabra = [stemmer.stem(w.lower()) for w in documento]
        for w in palabras:
            if w in auxPalabra:
                cubeta.append(1)
            else:
                cubeta.append(0)
        filaSalida = salidaVacia[:]
        filaSalida[tags.index(auxY[x])] = 1
        entrenamiento.append(cubeta)
        salida.append(filaSalida)

    entrenamiento = numpy.array(entrenamiento)
    salida = numpy.array(salida)
    with open("variables.pickle", "wb") as archivoPickle:
        pickle.dump((palabras, tags, entrenamiento, salida), archivoPickle)

tf.compat.v1.reset_default_graph()

red = tflearn.input_data(shape=[None, len(entrenamiento[0])])
red = tflearn.fully_connected(red, 40)
red = tflearn.fully_connected(red, 40)
red = tflearn.fully_connected(red, len(salida[0]), activation="softmax")
red = tflearn.regression(red)

modelo = tflearn.DNN(red)

modelo.fit(entrenamiento, salida, n_epoch=1000, batch_size=10, show_metric=True)
modelo.save("modelo.tflearn")

estado_creando_cita = False  # Estado inicial, no se está creando una cita

def chatbot_response(mensaje, tag_especifico=None):
    global estado_creando_cita
    cubeta = [0 for _ in range(len(palabras))]
    entradaProcesada = nltk.word_tokenize(mensaje)
    entradaProcesada = [stemmer.stem(palabra.lower()) for palabra in entradaProcesada]
    print("tag verid: ", tag_especifico)
    
    # Verificar si el usuario está en el proceso de creación de una cita
    if estado_creando_cita and tag_especifico != "citas_medicas":
        datos_entrenamiento = datos_citas["contenido"]
    else:
        datos_entrenamiento = datos["contenido"]
    
    # Verificar si la entrada es una fecha en formato AAAA-MM-DD
    if re.match(r'\d{4}-\d{2}-\d{2}', mensaje):
        respuesta = "Entendido, estás buscando una cita para la fecha proporcionada."
        # Lógica para proporcionar especialidades disponibles para la fecha
        
        # Modificar el tag_especifico según la condición
        
        tag = "Especialidades"
        
            
    else:
        for palabraIndividual in entradaProcesada:
            for i, palabra in enumerate(palabras):
                if palabra == palabraIndividual:
                    cubeta[i] = 1
        resultados = modelo.predict([numpy.array(cubeta)])
        resultadosIndices = numpy.argmax(resultados)
        tag = tags[resultadosIndices]
        print("prediccion:", resultados[0][resultadosIndices], tag)
        if resultados[0][resultadosIndices] > 0.45:
            respuesta_mostrada = False
            respuesta = ""
            for tagAux in datos_entrenamiento:
                if tagAux["tag"] == tag:
                    respuestas = tagAux["respuestas"]
                    respuesta = random.choice(respuestas)
                    respuesta_mostrada = True

            if not respuesta_mostrada:
                respuesta = "Proceso de creación de citas cancelado."
        else:
            respuesta = "No entendí, ¿puedes formular tu pregunta de nuevo?"
            tag = ""
    
    # Lógica para verificar si el usuario quiere crear o cancelar una cita
    if estado_creando_cita:
        print(tag_especifico)
        if tag == "cancelar_agendacion":
            respuesta = "Proceso de creación de citas cancelado."
            estado_creando_cita = False
        else:
            respuesta = "Por favor, proporciona la información necesaria para crear la cita."
    elif tag_especifico == "citas_medicas":
        estado_creando_cita = True
    
    print("resp:", respuesta)
    return respuesta, tag



def registrar_cita(especialidades,fecha_ingresada):
    respuesta = "Por favor, elige una de las siguientes especialidades:<br>"
    if len(especialidades) > 0:
       for i, especialidad in enumerate(especialidades, start=1):
            respuesta += f"{i}. {especialidad['especialidad']}<br>"
    else:
        respuesta = f"No hay especialidades disponibles para la fecha {fecha_ingresada}."

    return respuesta

def getMedicos(medicos):
    respuesta = "Por favor, elige una de las siguientes médicos:<br>"
    if len(medicos) > 0:
       for i, medico in enumerate(medicos, start=1):
            respuesta += f"{i}. {medico['medico_nombre']}<br>"
    else:
        respuesta = f"No hay médicos disponibles."

    return respuesta

def getHorario( horarios):
    respuesta = "Por favor, elige una de las siguientes Horarios disponibles:<br>"
    if len(horarios) > 0:
        for i, horario in enumerate(horarios, start=1):
            respuesta += f"{i}. {horario['NOM_HORA']}<br>"
    else:
        respuesta = f"No hay horarios disponibles."

    return respuesta

def getcitas( cita_Cancelada):
    respuesta = "Por favor, elige una de las siguientes citas que desea Anular o Cancelar :<br>"
    if len(cita_Cancelada) > 0:
        for i, cancelar in enumerate(cita_Cancelada, start=1):
            respuesta += f"{i}. Fecha: {cancelar['FECHA']},"
            respuesta += f"   Hora: {cancelar['NOM_HORA']},"
            respuesta += f"   Médico: {cancelar['nombre_medico']},"
            respuesta += f"   Especialidad: {cancelar['ESP_NOM']}"
            respuesta += "<br>"
    else:
        respuesta = f"No hay citas disponibles para cancelar."

    return respuesta


app = Flask(__name__)
CORS(app) 
especialidades = []
medicos=[]
horarios=[]
@app.route('/api/chatbot', methods=['POST'])
def api_chatbot():
    global especialidades
    global medicos
    global horarios
    global cita_Cancelada
    global estado_creando_cita
    datos_cita = {}
    if request.method == 'POST':

        
        try:
            mensaje = request.json['mensaje']  # Obtener el mensaje enviado desde la aplicación de Angular
            id_paciente_index = mensaje.find("Paciente ID: ")
            if id_paciente_index != -1:
                id_paciente = mensaje[id_paciente_index + len("Paciente ID: "):]
                
            else:
                id_paciente = "No se encontró el ID del paciente"

            # Buscar y obtener la fecha ingresada
            
            fecha_pattern = r'\d{4}-\d{2}-\d{2}'  # Expresión regular para buscar fechas en formato AAAA-MM-DD
            fechas_encontradas = re.findall(fecha_pattern, mensaje)
            if fechas_encontradas:
                fecha_ingresada = fechas_encontradas[0]  # Tomar la primera fecha encontrada
                print("Fecha ingresada:", fecha_ingresada)
            else:
                fecha_ingresada = "No se encontró fecha en el mensaje"
            datos_citas['id_paciente']=id_paciente
            print("ID del paciente:", id_paciente)
            print("Fecha ingresada:", fecha_ingresada)
            
            tag = chatbot_response(mensaje)[1]  # Obtener solo el tag de la respuesta del chatbot
            
            # Verificar si el tag es "citas_medicas" o "cancelar_cita"
            if tag in ["citas_medicas", "cancelar_cita"]:
                tag_especifico = tag
            else:
                tag_especifico = None
            respuesta, tag = chatbot_response(mensaje, tag_especifico) # Obtener la respuesta del chatbot
           
            if tag == 'Especialidades':
                # Obtener la lista de especialidades desde la API
                response_especialidades = requests.get(f'http://localhost/APIDISPENSARIO/listardoctoresespecialidad/{fecha_ingresada}')
                especialidades = response_especialidades.json()["info"]["items"]
                respuesta_registrar_cita = registrar_cita( especialidades, fecha_ingresada)
                # Almacenar la fecha_ingresada
                datos_citas['fecha_ingresada']=fecha_ingresada
                return jsonify({'respuesta': respuesta_registrar_cita, 'tag': tag})
            
            if tag == 'EleccionEspecialidad':
                # Buscar un número en el mensaje del usuario
                print("sms1: ", mensaje)
                numero_elegido = next((int(palabra) for palabra in mensaje.split() if palabra.isdigit()), None)
                print("sms1: ", numero_elegido)

                especialidad_elegida = None

                if numero_elegido is not None and 1 <= numero_elegido <= len(especialidades):
                    # Obtener la especialidad elegida según el número
                    print("esp: ", especialidades)

                    especialidad_elegida = especialidades[numero_elegido - 1]['especialidad']
                    print("esp: ", especialidad_elegida)

                else:
                    palabras_mensaje = mensaje.lower().split()
                    for especialidad in especialidades:
                        nombre_especialidad = especialidad['especialidad'].lower()
                        if any(palabra in nombre_especialidad for palabra in palabras_mensaje):
                            especialidad_elegida = especialidad['especialidad']
                            break
                            
                if especialidad_elegida:
                    fecha_ingresada=datos_citas.get('fecha_ingresada')
                    # Almacenar la especialidad_elegida 
                    
                    # Continuar con el proceso de creación de la cita y obtener la respuesta de continuación
                    #respuesta_continuacion = f"Has elegido la especialidad: {especialidad_elegida}"
                    response_Med = requests.get(f'http://localhost/APIDISPENSARIO/listardoctoresdisp/{fecha_ingresada}/{especialidad_elegida}')
                    medicos = response_Med.json()["info"]["items"]
                    respuesta_Med = getMedicos( medicos)
                else:
                    respuesta_Med = "No se reconoció ninguna especialidad en tu mensaje. Por favor, elige una especialidad válida."

                return jsonify({'respuesta': respuesta_Med, 'tag': tag})

            if tag == 'EleccionMedicos':
                numero_elegido = next((int(palabra) for palabra in mensaje.split() if palabra.isdigit()), None)
                medico_elegido = None

                if numero_elegido is not None and 1 <= numero_elegido <= len(medicos):
                    # Obtener el medico elegido según el número
                    print("esp: ", medicos)

                    medico_elegido = medicos[numero_elegido - 1]['MED_ID']
                    print("esp: ", medico_elegido)

                else:
                    palabras_mensaje = mensaje.lower().split()
                    for medico in medicos:
                        nombre_medico = medico['medico_nombre'].lower()
                        if any(palabra in nombre_medico for palabra in palabras_mensaje):
                            medico_elegido = medico['MED_ID']
                            break
                
                if medico_elegido:
                    fecha_ingresada=datos_citas.get('fecha_ingresada')
                    # Almacenar la especialidad_elegida 
                    datos_citas['medico_elegido']=medico_elegido
                    # Continuar con el proceso de creación de la cita y obtener la respuesta de continuación
                    #respuesta_Med = f"Has elegido el médico: {medico_elegido}"
                    response_Med = requests.get(f'http://localhost/APIDISPENSARIO/listarhoraxFecha/{fecha_ingresada}/{medico_elegido}')
                    horarios = response_Med.json()["info"]["items"]
                    respuesta_Med = getHorario( horarios)
                else:
                    respuesta_Med = "No se reconoció ningun médico en tu mensaje. Por favor, elige un médico válido."

                return jsonify({'respuesta': respuesta_Med, 'tag': tag})

            if tag == 'EleccionHorarios':
                numero_elegido = next((int(palabra) for palabra in mensaje.split() if palabra.isdigit()), None)
                horario_elegido = None

                if numero_elegido is not None and 1 <= numero_elegido <= len(horarios):
                    # Obtener el medico elegido según el número
                    print("esp: ", horarios)

                    horario_elegido = horarios[numero_elegido - 1]['HORA_ID']
                    print("esp: ", horario_elegido)
                else:
                    palabras_mensaje = mensaje.lower().split()
                    for horario in horarios:
                        nombre_horario = horario['NOM_HORA'].lower()
                        if any(palabra in nombre_horario for palabra in palabras_mensaje):
                            horario_elegido = horario['HORA_ID']
                            break
                
                if horario_elegido:
                    datos_citas['horario_elegido']=horario_elegido

                    try:
                        print("pac_id",datos_citas['id_paciente'])
                        print("med_id",datos_citas['medico_elegido'])
                        print("fecha",datos_citas['fecha_ingresada'])
                        print("hora_id",datos_citas['horario_elegido'])
                        response_cita = requests.post(f'http://localhost/APIDISPENSARIO/creaCita', data={
                            'pac_id': datos_citas['id_paciente'],
                            'med_id': datos_citas['medico_elegido'],
                            'fecha': datos_citas['fecha_ingresada'],
                            'hora_id': datos_citas['horario_elegido']
                        })

                        if response_cita.status_code == 200:
                            respuesta_hora= response_cita.json()["mensaje"]
                            print("cita", respuesta_hora)
                            #respuesta_hora = "Cita creada exitosamente."
                            estado_creando_cita = False
                        else:
                            respuesta_hora = "Hubo un problema al crear la cita."
                    except requests.exceptions.RequestException as e:
                        respuesta_hora = "Ocurrió un error al intentar crear la cita. Por favor, inténtalo nuevamente más tarde."

                else:
                    respuesta_hora = "No se reconoció ningun horario en tu mensaje. Por favor, elige un horario válido."

                return jsonify({'respuesta': respuesta_hora, 'tag': tag})
            
            if tag == 'cancelar_cita':
                mensaje = request.json['mensaje'] 
                fecha_ingresada=datos_citas.get('fecha_ingresada')
                print("id paciente", id_paciente)
                print("fecha", fecha_ingresada)
                response_Cancelar = requests.get(f'http://localhost/APIDISPENSARIO/listarcitasxID/{id_paciente}')
                cita_Cancelada = response_Cancelar.json()["info"]["items"]
                resp_cita = getcitas( cita_Cancelada)
                if resp_cita != "No hay citas disponibles para cancelar.":
                    return jsonify({'respuesta': resp_cita, 'tag': tag})
                else:
                    return jsonify({'respuesta': resp_cita})
            
            if tag == 'EleccionCita':
                numero_elegido = next((int(palabra) for palabra in mensaje.split() if palabra.isdigit()), None)
                cita_elegida = None
                print("number: ",cita_Cancelada)
                
                if numero_elegido is not None and 1 <= numero_elegido <= len(cita_Cancelada):
                    # Obtener el medico elegido según el número
                    print("esp: ", cita_Cancelada)

                    cita_elegida = cita_Cancelada[numero_elegido - 1]
                    print("esp: ", cita_elegida)

                else:
                    # Extraer palabras clave del mensaje
                    
                    palabras_clave = re.findall(r'\w+', mensaje.lower())
                    print("palabra sms: ",palabras_clave)
                     # Buscar coincidencias con detalles de la cita
                    for cita in cita_Cancelada:
                        detalles_cita = f"{cita['FECHA']} {cita['nombre_medico']} {cita['ESP_NOM']} {cita['NOM_HORA']}".lower()
                        coincidencias = 0
                        for palabra in palabras_clave:
                            if palabra in detalles_cita:
                                coincidencias += 1
                        if coincidencias > 1:
                            cita_elegida = cita
                            break
                    print("cita elegida: ",cita_elegida)        
                if cita_elegida:
                    cita_id = cita_elegida['CITAS_ID']
                    nuevo_estado = {
                        "med_id": cita_elegida['MED_ID'],
                        "fecha": cita_elegida['FECHA'],
                        "hora_id": cita_elegida['HORA_ID'],
                        "estado": "Anulado"
                    }
                    print("cita: ",nuevo_estado)
                    try:
                        respo_cita = requests.post(f'http://localhost/APIDISPENSARIO/actualizarcitap/{cita_id}', data=nuevo_estado)
                        print("cita elegida: ",respo_cita.json())
                        if respo_cita.status_code == 200:
                            print("La cita se ha actualizado con éxito.")
                            resp_cita='Cita Cancelada con éxito'
                        else:
                            print("No se pudo actualizar la cita. Código de estado:", respo_cita.status_code)
                            resp_cita='No se pudo anular la cita'
                    except requests.exceptions.RequestException as e:
                        print("Error al realizar la solicitud:", e)
                else:
                    resp_cita="algo salio mal"

                return jsonify({'respuesta': resp_cita, 'tag': tag})
                 



            return jsonify({'respuesta': respuesta, 'tag': tag})  # Devolver la respuesta como JSON
        except Exception as e:
            return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run()
