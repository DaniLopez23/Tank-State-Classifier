import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

DATA_STRATEGY = "second"   # "second" o "minute"
DATE = "2024-09-27"


def main(csv_path):
    # Leer y validar CSV
    try:
        df = pd.read_csv(csv_path, parse_dates=["DateTime"])
        required_columns = ["DateTime", "AccelX", "Surface temperature (ºC)", "Over surface temperature (ºC)"]
        
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"Columnas faltantes: {missing}")
            
    except Exception as e:
        print(f"Error leyendo CSV: {str(e)}")
        return

    # Preparar datos para la API
    data = {
        "AccelX": df["AccelX"].fillna(0).tolist(),
        "SurfaceTemperature": df["Surface temperature (ºC)"].fillna(0).tolist(),
        "OverSurfaceTemperature": df["Over surface temperature (ºC)"].fillna(0).tolist()
    }
    
    # Hacer la petición
    try:
        response = requests.post("http://localhost:8000/predict", json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error en la petición: {str(e)}")
        if response:
            print(f"Respuesta del servidor: {response.text}")
        return

    # Procesar resultados
    if "windows" not in result:
        print("Respuesta inesperada del servidor:")
        print(result)
        return
    
    # Crear DataFrame con predicciones
    predictions_df = pd.DataFrame(result["windows"])
    
    # Mapear predicciones a los datos originales
    df["Prediction"] = "UNKNOWN"
    
    for _, window in predictions_df.iterrows():
        try:
            start_idx = int(window['start'])
            end_idx = int(window['end'])
            start_idx = max(0, min(start_idx, len(df)-1))
            end_idx = max(0, min(end_idx, len(df)-1))
            df.loc[start_idx:end_idx, "Prediction"] = window['prediction']
        except (KeyError, ValueError) as e:
            print(f"Error procesando ventana: {str(e)}")
    
    # Manejar solapamientos y guardar CSV
    df["Prediction"] = df["Prediction"].ffill().bfill().fillna("UNKNOWN")
    output_path = "predictions.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Predicciones guardadas en: {output_path}")
    
    # Visualización
    plot_results(df)

def plot_results(df):
    # Configurar figura
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Graficar datos brutos
    ax.plot(df['DateTime'], df['AccelX'], label='Acelerómetro X', color='blue', alpha=0.6)
    
    # Manejo de colores actualizado
    unique_states = df["Prediction"].unique()
    colors = plt.get_cmap('tab10', len(unique_states))  # Línea corregida
    
    # Dibujar regiones de predicción
    current_state = None
    start_time = None
    
    for idx, row in df.iterrows():
        if row["Prediction"] != current_state:
            if current_state is not None:
                ax.axvspan(start_time, row["DateTime"], alpha=0.3, 
                          color=colors(list(unique_states).index(current_state)), 
                          label=current_state)
            current_state = row["Prediction"]
            start_time = row["DateTime"]
    
    if current_state is not None:
        ax.axvspan(start_time, df["DateTime"].iloc[-1], alpha=0.3,
                  color=colors(list(unique_states).index(current_state)),
                  label=current_state)
    
    # Configurar ejes y leyenda
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Aceleración X")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Estados", 
             bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title(f"Predicciones - {df['DateTime'].dt.date.iloc[0]}")
    plt.tight_layout()
    plt.savefig("predicciones.png", bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    csv_test_path = "test.csv"
    csv_path =f"data_per_{DATA_STRATEGY}_strategy/merged_data/merged_data_{DATE}.csv"
    main(csv_path)