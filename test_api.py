import requests
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def main(csv_path):
    # Leer el CSV
    df = pd.read_csv(csv_path, parse_dates=["DateTime"], infer_datetime_format=True)
    # AccelX,DateTime,Over surface temperature (ºC),Surface temperature (ºC)
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
    result = response.json()
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    # Crear DataFrame con resultados
    df["Prediction"] = result["predictions"]
    
    # Crear figura
    plt.figure(figsize=(15, 10))
    
    # Configurar eje de tiempo
    x_axis = df["DateTime"] if "DateTime" in df.columns else df.index
    
    # Gráfica de sensores
    plt.subplot(4, 1, 1)
    plt.plot(x_axis, df["AccelX"], label='AccelX')
    plt.title("Datos de Sensores y Predicciones")
    plt.ylabel("AccelX")
    plt.legend()
    
    plt.subplot(4, 1, 2)
    plt.plot(x_axis, df["Surface temperature (ºC)"], color='orange', label='Surface Temp')
    plt.ylabel("Temp. Superficie (ºC)")
    plt.legend()
    
    plt.subplot(4, 1, 3)
    plt.plot(x_axis, df["Over surface temperature (ºC)"], color='red', label='Over Surface Temp')
    plt.ylabel("Temp. Exterior (ºC)")
    plt.legend()
    
    # Gráfica de predicciones
    plt.subplot(4, 1, 4)
    unique_states = list(set(result["predictions"]))
    color_map = {state: plt.cm.tab10(i) for i, state in enumerate(unique_states)}
    
    for state in unique_states:
        mask = df["Prediction"] == state
        plt.scatter(x_axis[mask], df["Prediction"][mask], 
                    color=color_map[state], label=state, alpha=0.7)
    
    plt.yticks(unique_states, unique_states)
    plt.ylabel("Estado")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    csv_path = "test.csv"
    
    main(csv_path)