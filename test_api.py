import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse

def main(csv_path):
    # Leer el CSV
    df = pd.read_csv(csv_path, parse_dates=["DateTime"])
    
    # Verificar columnas requeridas
    required_columns = ["AccelX", "Surface temperature (ºC)", "Over surface temperature (ºC)"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"El CSV debe contener las columnas: {required_columns}")
    
    # Preparar datos para la API
    data = {
        "AccelX": df["AccelX"].tolist(),
        "SurfaceTemperature": df["Surface temperature (ºC)"].tolist(),
        "OverSurfaceTemperature": df["Over surface temperature (ºC)"].tolist()
    }
    
    # Hacer la petición
    response = requests.post("http://localhost:8000/predict", json=data)

    # Verificar el contenido de la respuesta
    print("Status Code:", response.status_code)
    print("Response Text:", response.text)

    # Intentar decodificar la respuesta como JSON
    try:
        result = response.json()
    except requests.exceptions.JSONDecodeError as e:
        print(f"Error al decodificar JSON: {e}")
        return
    
    # Agregar predicciones al DataFrame
    df["Prediction"] = result["predictions"]
    
    # Crear figura y ejes
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Graficar datos de IMU en ax1
    ax1.plot(df['DateTime'], df['AccelX'], label='Accel X', color='b', marker='.', linestyle='-', markersize=4)
    ax1.set_xlabel("Fecha/Hora")
    ax1.set_ylabel("Accel X", color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True)
    
    # Crear un segundo eje para las temperaturas
    ax2 = ax1.twinx()
    ax2.plot(df['DateTime'], df['Surface temperature (ºC)'], label='Surface Temp', color='g', marker='.', linestyle='-', markersize=4)
    ax2.plot(df['DateTime'], df['Over surface temperature (ºC)'], label='Over Surface Temp', color='orange', marker='.', linestyle='-', markersize=4)
    ax2.set_ylabel("Temperatura (ºC)", color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    
    # Colorear las zonas con las etiquetas en el gráfico
    unique_states = list(set(df["Prediction"]))
    color_map = {
        "MAINTENANCE": 'lightyellow',
        "CLEANING": 'orange',
        "MILKING": 'lightgreen',
        "COOLING": 'lightblue',
        "EMPTY TANK": 'lightcoral'
    }
    
    for state in unique_states:
        if state not in color_map:
            continue  # Ignorar estados desconocidos
        
        mask = df["Prediction"] == state
        ax1.fill_between(df['DateTime'], ax1.get_ylim()[0], ax1.get_ylim()[1], 
                         where=mask, color=color_map[state], alpha=0.3, label=f'{state} Period')
    
    # Formatear el eje X
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.xticks(rotation=45)
    
    # Agregar leyendas
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc='upper left')
    ax2.legend(lines2, labels2, loc='upper right')
    
    # Título y mostrar gráfico
    plt.title("Datos IMU, Temperaturas y Predicciones en el Tiempo")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    csv_path = "test.csv"
    main(csv_path)
