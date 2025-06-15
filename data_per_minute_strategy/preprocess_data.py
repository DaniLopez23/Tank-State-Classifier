import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

DATE = "2024-08-19"

# Definir rutas de los archivos
TEMP_FILE = f"../temp_data/temp_data_{DATE}.csv"
IMU_FILE = f"../accel_data/accel_data_{DATE}.csv"
OUTPUT_FILE = f"merged_data/merged_data_{DATE}.csv"

# Valores estándar
STANDARD_INIT_SUBMERGED_TEMP = -127.0
STANDARD_INIT_OVER_SURFACE_TEMP = 6.5
STANDARD_INIT_SURFACE_TEMP = 4.31

def merge_data(temp_file, imu_file, output_file):
    # Cargar los archivos CSV
    temp_df = pd.read_csv(temp_file)
    imu_df = pd.read_csv(imu_file)
    
    # Convertir la columna DateTime a formato datetime
    temp_df["DateTime"] = pd.to_datetime(temp_df["_time"])
    imu_df["DateTime"] = pd.to_datetime(imu_df["_time"])

    # Definir el rango de tiempo basado en los datos
    start_time = min(temp_df["DateTime"].min(), imu_df["DateTime"].min())
    end_time = max(temp_df["DateTime"].max(), imu_df["DateTime"].max())

    # Crear un DataFrame con timestamps cada minuto
    time_index = pd.date_range(start=start_time, end=end_time, freq="1T")  # 1T = 1 minuto
    merged_df = pd.DataFrame({"DateTime": time_index})

    # Unir datasets usando la fecha más cercana
    temp_df = temp_df.set_index("DateTime").reindex(time_index, method="nearest").reset_index().rename(columns={"index": "DateTime"})
    imu_df = imu_df.set_index("DateTime").reindex(time_index, method="nearest").reset_index().rename(columns={"index": "DateTime"})

    # Fusionar los datos
    merged_df = merged_df.merge(temp_df, on="DateTime", how="left")
    merged_df = merged_df.merge(imu_df, on="DateTime", how="left")

    # Rellenar valores NaN con valores estándar
    for col, std_val in {
        "Surface temperature (ºC)": STANDARD_INIT_SURFACE_TEMP,
        "Over surface temperature (ºC)": STANDARD_INIT_OVER_SURFACE_TEMP,
        "Submerged temperature (ºC)": STANDARD_INIT_SUBMERGED_TEMP
    }.items():
        merged_df[col] = merged_df[col].fillna(std_val).ffill().bfill()  # Relleno hacia adelante y atrás

    # Eliminar columnas innecesarias
    merged_df.drop(columns=["_time_x", "_time_y", "Submerged temperature (ºC)"], errors="ignore", inplace=True)
    
    # Guardar el resultado en un archivo CSV
    merged_df.to_csv(output_file, index=False)
    print(f"Archivo combinado guardado en: {output_file}")
    
    return merged_df


def plot_data(merged_df):
    """Genera un gráfico con bandas para ventanas de 20 minutos."""
    
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Graficar AccelX en el eje primario (IMU)
    ax1.plot(merged_df["DateTime"], merged_df["AccelX"], label="AccelX", color='b', marker='.', linestyle='-', markersize=4)
    ax1.set_xlabel("Fecha/Hora")
    ax1.set_ylabel("AccelX", color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True)

    # Crear un segundo eje para las temperaturas
    ax2 = ax1.twinx()
    ax2.plot(merged_df["DateTime"], merged_df["Surface temperature (ºC)"], label="Surface Temp", color='g', marker='.', linestyle='-', markersize=4)
    ax2.plot(merged_df["DateTime"], merged_df["Over surface temperature (ºC)"], label="Over Surface Temp", color='orange', marker='.', linestyle='-', markersize=4)
    ax2.set_ylabel("Temperatura (ºC)", color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    
    # Formatear el eje X para mostrar horas y minutos con marcas cada 10 minutos
    ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=60))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    plt.xticks(rotation=45)
    
    # Agregar leyendas separadas
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    ax1.legend(lines1, labels1, loc='upper left')
    ax2.legend(lines2, labels2, loc='upper right')
    
    plt.title("Datos IMU y Temperatura con Ventanas de 20 Minutos")
    plt.tight_layout()
    plt.show()


# Ejecutar la función
merged_data = merge_data(TEMP_FILE, IMU_FILE, OUTPUT_FILE)
plot_data(merged_data)
