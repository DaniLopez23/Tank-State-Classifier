import csv
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import json  # Importar para leer el archivo JSON

# --- CONSTANTES A CONFIGURAR ---
DATE = "2024-08-24"  # La fecha que quieres procesar

INPUT_CSV = f"merged_data/merged_data_{DATE}.csv"  
OUTPUT_CSV = f"labeled_data/labeled_data_{DATE}.csv"
INTERVALS_FILE = "labelFile.json"  # Archivo JSON con los intervalos

COLUMNA_TIMESTAMP = 'DateTime'
FORMATO_TIMESTAMP = '%Y-%m-%d %H:%M:%S%z'  # Ahora incluye la zona horaria

# --- CARGAR INTERVALOS DESDE ARCHIVO JSON ---
def cargar_intervalos_desde_json(fecha):
    with open(INTERVALS_FILE, 'r') as archivo_json:
        intervalos_data = json.load(archivo_json)
        # Devolver los intervalos para la fecha solicitada
        return intervalos_data.get(fecha, [])

INTERVALOS_ETIQUETAS = cargar_intervalos_desde_json(DATE)

# --- PROCESAMIENTO ---
# Convertir intervalos a datetime con zona horaria
intervalos = []
for intervalo in INTERVALOS_ETIQUETAS:
    intervalo_convertido = {
        'inicio': datetime.strptime(intervalo['inicio'], FORMATO_TIMESTAMP),
        'fin': datetime.strptime(intervalo['fin'], FORMATO_TIMESTAMP),
        'etiqueta': intervalo['etiqueta']
    }
    intervalos.append(intervalo_convertido)

# Leer y procesar archivo
with open(INPUT_CSV, 'r') as archivo_entrada:
    lector = csv.DictReader(archivo_entrada)
    filas = list(lector)
    
    campos = lector.fieldnames + ['ETIQUETA']

    for fila in filas:
        # Parsear timestamp con zona horaria
        timestamp = datetime.strptime(fila[COLUMNA_TIMESTAMP], FORMATO_TIMESTAMP)
        fila['ETIQUETA'] = ''
        
        for intervalo in intervalos:
            if intervalo['inicio'] <= timestamp <= intervalo['fin']:
                fila['ETIQUETA'] = intervalo['etiqueta']
                break

# Escribir resultados
with open(OUTPUT_CSV, 'w', newline='') as archivo_salida:
    escritor = csv.DictWriter(archivo_salida, fieldnames=campos)
    escritor.writeheader()
    escritor.writerows(filas)

print(f"Proceso completado. Resultados en: {OUTPUT_CSV}")

# --- GRAFICADO ---
def graficar_datos_con_etiquetas(labeled_data, intervalos):
    """
    Función para graficar datos etiquetados con dos ejes (uno para IMU y otro para las temperaturas),
    y marcar las zonas de eventos.
    """
    # Crear gráfico
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Graficar datos de IMU (en el eje primario)
    ax1.plot(labeled_data['DateTime'], labeled_data['AccelX'], label='Accel X', color='b', marker='.', linestyle='-', markersize=4)
    ax1.set_xlabel("Fecha/Hora")
    ax1.set_ylabel("Accel X", color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True)

    # Crear un segundo eje para las temperaturas
    ax2 = ax1.twinx()
    ax2.plot(labeled_data['DateTime'], labeled_data['Surface temperature (ºC)'], label='Surface Temp', color='g', marker='.', linestyle='-', markersize=4)
    ax2.plot(labeled_data['DateTime'], labeled_data['Over surface temperature (ºC)'], label='Over Surface Temp', color='orange', marker='.', linestyle='-', markersize=4)
    ax2.set_ylabel("Temperatura (ºC)", color='g')
    ax2.tick_params(axis='y', labelcolor='g')

    # Colorear las zonas con las etiquetas en el gráfico
    for intervalo in intervalos:
        start_time = intervalo['inicio']
        end_time = intervalo['fin']
        label = intervalo['etiqueta']

        # Elegir un color basado en la etiqueta
        if label == 'MAINTENANCE':
            color = 'lightyellow'
        elif label == 'CLEANING':
            color = 'orange'
        elif label == 'MILKING':
            color = 'lightgreen'
        elif label == 'COOLING':
            color = 'lightblue'
        elif label == "EMPTY TANK":
            color = 'lightcoral'
        else:
            color = 'lightgrey'

        # Marcar la zona con un color en el gráfico
        ax1.axvspan(start_time, end_time, color=color, alpha=0.5, label=f'{label} Period')

    # Formatear el eje X para mostrar horas y minutos
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # Formato para mostrar horas y minutos
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))  # Mostrar una marca cada hora
    plt.xticks(rotation=45)

    # Agregar leyendas separadas para los dos ejes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    # Leyenda para el eje IMU
    ax1.legend(lines1, labels1, loc='upper left')

    # Leyenda para el eje de temperatura
    ax2.legend(lines2, labels2, loc='upper right')

    # Título del gráfico
    plt.title("Datos IMU, Temperaturas y Etiquetas de Eventos en el Tiempo")

    # Ajustar el layout
    plt.tight_layout()

    # Mostrar el gráfico
    plt.show()

# Cargar datos etiquetados
labeled_data = pd.read_csv(OUTPUT_CSV)
labeled_data['DateTime'] = pd.to_datetime(labeled_data['DateTime'], format=FORMATO_TIMESTAMP)

# Llamar a la función para graficar
graficar_datos_con_etiquetas(labeled_data, intervalos)
