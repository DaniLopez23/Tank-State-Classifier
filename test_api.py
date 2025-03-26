import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

DATE = "2024-08-25"
DATA_STRATEGY = "5_second"

def main(csv_path):
    # Leer el CSV
    df = pd.read_csv(csv_path, parse_dates=["DateTime"])
    
    # Preparar datos para la API
    request_data = {
        "data": [
            {
                "DateTime": row["DateTime"].strftime("%Y-%m-%d %H:%M:%S%z"),
                "AccelX": row["AccelX"],
                "OverSurfaceTemperature": row["Over surface temperature (ºC)"],
                "SurfaceTemperature": row["Surface temperature (ºC)"]
            }
            for _, row in df.iterrows()
        ]
    }
    
    # Hacer la petición
    response = requests.post("http://localhost:8000/predict", json=request_data)

    # Manejar la respuesta
    if response.status_code != 200:
        print(f"Error en la petición: {response.status_code}")
        print(response.text)
        return
    
    result = response.json()
    print("Respuesta recibida:", result)

    # En la sección de procesamiento de intervalos:
    intervals = result.get('intervals', [])
    states = result.get('states', [])

    # Convertir fechas del DataFrame a datetime sin timezone
    df['DateTime'] = pd.to_datetime(df['DateTime']).dt.tz_localize(None)

    # Mapear cada intervalo al DataFrame
    for interval in intervals:
        try:
            # Convertir strings a datetime objects (UTC)
            start = pd.to_datetime(interval['inicio'], format='%d/%m/%Y %H:%M:%S').tz_localize('UTC').tz_convert(None)
            end = pd.to_datetime(interval['fin'], format='%d/%m/%Y %H:%M:%S').tz_localize('UTC').tz_convert(None)
            estado = interval['estado']
            
            # Crear máscara de tiempo
            mask = (df['DateTime'] >= start) & (df['DateTime'] <= end)
            df.loc[mask, 'Prediction'] = estado
        except Exception as e:
            print(f"Error procesando intervalo {interval}: {str(e)}")

    # Crear figura y ejes
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Graficar datos de IMU
    ax1.plot(df['DateTime'], df['AccelX'], label='Accel X', color='b', marker='.', linestyle='-', markersize=4)
    ax1.set_xlabel("Fecha/Hora")
    ax1.set_ylabel("Accel X", color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True)
    
    # Crear eje secundario para temperaturas
    ax2 = ax1.twinx()
    ax2.plot(df['DateTime'], df['Surface temperature (ºC)'], label='Surface Temp', color='g', marker='.', linestyle='-', markersize=4)
    ax2.plot(df['DateTime'], df['Over surface temperature (ºC)'], label='Over Surface Temp', color='orange', marker='.', linestyle='-', markersize=4)
    ax2.set_ylabel("Temperatura (ºC)", color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    
    # Colorear zonas según estados
    color_map = {
        "MAINTENANCE": 'yellow',
        "CLEANING": 'orange',
        "MILKING": 'blue',
        "COOLING": 'green',
        "EMPTY TANK": 'red',
        "UNKNOWN": 'gray'
    }
    
    # Crear áreas para cada estado
    prev_time = None
    current_state = None
    
    for _, row in df.iterrows():
        if row['Prediction'] != current_state:
            if prev_time is not None:
                ax1.axvspan(prev_time, row['DateTime'], facecolor=color_map.get(current_state, 'gray'), alpha=0.3)
            current_state = row['Prediction']
            prev_time = row['DateTime']
    
    # Último segmento
    if prev_time is not None:
        ax1.axvspan(prev_time, df['DateTime'].iloc[-1], facecolor=color_map.get(current_state, 'gray'), alpha=0.3)
    
    # Leyenda personalizada
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[state], alpha=0.3, label=state) 
                      for state in states + ['UNKNOWN'] if state in color_map]
    
    ax1.legend(handles=legend_elements, loc='upper left')
    
    # Formatear eje X
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.xticks(rotation=45)
    
    # Título y mostrar
    plt.title("Datos de sensores y estados predichos")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_csv_path = "test.csv"
    csv_path = f"data_per_{DATA_STRATEGY}_strategy/merged_data/merged_data_{DATE}.csv"
    main(csv_path)