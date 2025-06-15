import csv
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import json

# --- CONSTANTES A CONFIGURAR ---
DATE = "2024-08-19"
INPUT_CSV = f"merged_data/merged_data_{DATE}.csv"  
OUTPUT_CSV = f"labeled_data/labeled_data_{DATE}.csv"
INTERVALS_FILE = "../labelFile.json"

COLUMNA_TIMESTAMP = 'DateTime'
FORMATO_TIMESTAMP = '%Y-%m-%d %H:%M:%S%z'

# --- CARGAR INTERVALOS DESDE ARCHIVO JSON ---
def cargar_intervalos_desde_json(fecha):
    with open(INTERVALS_FILE, 'r') as archivo_json:
        intervalos_data = json.load(archivo_json)
    return intervalos_data.get(fecha, [])

INTERVALOS_ETIQUETAS = cargar_intervalos_desde_json(DATE)

# Convertir intervalos a datetime con zona horaria
intervalos = [
    {
        'inicio': datetime.strptime(intervalo['inicio'], FORMATO_TIMESTAMP),
        'fin': datetime.strptime(intervalo['fin'], FORMATO_TIMESTAMP),
        'etiqueta': intervalo['etiqueta']
    }
    for intervalo in INTERVALOS_ETIQUETAS
]

# Leer y procesar archivo
with open(INPUT_CSV, 'r') as archivo_entrada:
    lector = csv.DictReader(archivo_entrada)
    filas = list(lector)
    campos = lector.fieldnames + ['ETIQUETA']

    for fila in filas:
        timestamp = datetime.strptime(fila[COLUMNA_TIMESTAMP], FORMATO_TIMESTAMP)
        etiqueta_asignada = 'UNKNOWN'
        
        for intervalo in intervalos:
            if intervalo['inicio'] <= timestamp <= intervalo['fin']:
                etiqueta_asignada = intervalo['etiqueta']
                break
        
        fila['ETIQUETA'] = etiqueta_asignada

# Escribir resultados
with open(OUTPUT_CSV, 'w', newline='') as archivo_salida:
    escritor = csv.DictWriter(archivo_salida, fieldnames=campos)
    escritor.writeheader()
    escritor.writerows(filas)

print(f"Proceso completado. Resultados en: {OUTPUT_CSV}")

# --- GRAFICADO ---
def graficar_datos_con_etiquetas(labeled_data, intervalos):
    fig, ax1 = plt.subplots(figsize=(14, 8))

    ax1.plot(labeled_data['DateTime'], labeled_data['AccelX'], label='Accel X', color='b', marker='.', linestyle='-', markersize=4)
    ax1.set_xlabel("Fecha/Hora")
    ax1.set_ylabel("Accel X", color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(labeled_data['DateTime'], labeled_data['Surface temperature (ºC)'], label='Surface Temp', color='g', marker='.', linestyle='-', markersize=4)
    ax2.plot(labeled_data['DateTime'], labeled_data['Over surface temperature (ºC)'], label='Over Surface Temp', color='orange', marker='.', linestyle='-', markersize=4)
    ax2.set_ylabel("Temperatura (ºC)", color='g')
    ax2.tick_params(axis='y', labelcolor='g')

    for intervalo in intervalos:
        start_time = intervalo['inicio']
        end_time = intervalo['fin']
        label = intervalo['etiqueta']

        color_map = {
            'MAINTENANCE': 'lightyellow',
            'CLEANING': 'orange',
            'MILKING': 'lightgreen',
            'COOLING': 'lightblue',
            'EMPTY TANK': 'lightcoral'
        }
        color = color_map.get(label, 'lightgrey')
        ax1.axvspan(start_time, end_time, color=color, alpha=0.5, label=f'{label} Period')

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.xticks(rotation=45)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc='upper left')
    ax2.legend(lines2, labels2, loc='upper right')
    
    plt.title("Datos IMU, Temperaturas y Etiquetas de Eventos en el Tiempo")
    plt.tight_layout()
    plt.show()

# Cargar datos etiquetados
labeled_data = pd.read_csv(OUTPUT_CSV)
labeled_data['DateTime'] = pd.to_datetime(labeled_data['DateTime'], format=FORMATO_TIMESTAMP)

# Eliminar valores nulos
labeled_data.fillna('UNKNOWN', inplace=True)

graficar_datos_con_etiquetas(labeled_data, intervalos)