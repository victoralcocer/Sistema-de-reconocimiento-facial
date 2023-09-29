from cv2 import cv2
import face_recognition as fr
import os
import numpy
from datetime import datetime

#base de datos
carpeta = 'empleados'
imagenes = []
nombres_empleados = []
array_empleados = os.listdir(carpeta)

for nombre in array_empleados:
    imagen_actual = cv2.imread(f'{carpeta}/{nombre}')
    imagenes.append(imagen_actual)
    nombres_empleados.append(os.path.splitext(nombre)[0])

print(nombres_empleados)

#codificar imagenes
def codificar(imagenes):

    array_codificado = []
    for imagen in imagenes:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

        codificado = fr.face_encodings(imagen)[0]
        array_codificado.append(codificado)

    return array_codificado

#registrar ingresos
def registrar_ingresos(persona):
    archivo = open('registro.csv', 'r+')
    lista_datos = archivo.readlines()
    nombres_registros = []

    for linea in lista_datos:
        ingreso = linea.split(' ')
        nombres_registros.append(ingreso[0])

    if persona not in nombres_registros:
        fecha = datetime.now()
        string_fecha = fecha.strftime('%Y-%m-%d %H:%M:%S')
        archivo.writelines(f'\n{persona} {string_fecha}')


array_empleados_codificados = codificar(imagenes)

#encender la camara con cv2
captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)
exito, imagen = captura.read()

#comprobar si la foto se he realizado
if not exito:
    print("No se ha podido realizar la captura")
else:
    cara_captura = fr.face_locations(imagen)

    cara_captura_codificada = fr.face_encodings(imagen, cara_captura)
    #buscar coincidencias
    for caracodificada, caraubic in zip(cara_captura_codificada, cara_captura):
        coincidencias = fr.compare_faces(array_empleados_codificados, caracodificada)
        diferencias = fr.face_distance(array_empleados_codificados, caracodificada)

        print(diferencias)

        indice_coincidencia = numpy.argmin(diferencias)

        #coincidencias
        if diferencias[indice_coincidencia] > 0.6:
            print("No coincide con ninguno de los empleados")

        else:
            nombre=nombres_empleados[indice_coincidencia]
            print(f"Eres {nombre}, ¡Bienvenido al trabajo!")
            registrar_ingresos(nombre)
