import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DATE = "2024-08-17"

# Definir rutas de los archivos
TEMP_FILE = f"../temp_data/temp_data_{DATE}.csv"
IMU_FILE = f"../accel_data/accel_data_{DATE}.csv"
OUTPUT_FILE = f"merged_data/merged_data_{DATE}.csv"

# Nuevos valores estándar
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
    
    # Crear un rango completo de timestamps de un segundo
    start_time = min(temp_df["DateTime"].min(), imu_df["DateTime"].min())
    end_time = max(temp_df["DateTime"].max(), imu_df["DateTime"].max())
    full_range = pd.date_range(start=start_time, end=end_time, freq='s')  # Cambiado 'S' por 's'
    
    # Crear DataFrame con el rango de tiempo
    full_df = pd.DataFrame({"DateTime": full_range})
    
    # Fusionar los datos por timestamp
    merged_df = pd.merge_asof(full_df, imu_df.sort_values("DateTime"), on="DateTime", direction="backward")
    merged_df = pd.merge_asof(merged_df, temp_df.sort_values("DateTime"), on="DateTime", direction="backward")
    
    # Reemplazar el primer valor de temperatura si es NaN
    merged_df.loc[0, "Surface temperature (ºC)"] = merged_df.loc[0, "Surface temperature (ºC)"] if pd.notna(merged_df.loc[0, "Surface temperature (ºC)"]) else STANDARD_INIT_SURFACE_TEMP
    merged_df.loc[0, "Over surface temperature (ºC)"] = merged_df.loc[0, "Over surface temperature (ºC)"] if pd.notna(merged_df.loc[0, "Over surface temperature (ºC)"]) else STANDARD_INIT_OVER_SURFACE_TEMP
    merged_df.loc[0, "Submerged temperature (ºC)"] = merged_df.loc[0, "Submerged temperature (ºC)"] if pd.notna(merged_df.loc[0, "Submerged temperature (ºC)"]) else STANDARD_INIT_SUBMERGED_TEMP

    # Lógica para rellenar valores vacíos
    for col in ["Surface temperature (ºC)", "Over surface temperature (ºC)", "Submerged temperature (ºC)"]:
        merged_df[col] = merged_df[col].ffill().bfill()  # Cambiado `fillna(method=...)` por `ffill()` y `bfill()`
    
    # Eliminar columnas innecesarias
    merged_df = merged_df.drop(columns=["_time_x", "_time_y", "Submerged temperature (ºC)"], errors="ignore")
    
    # Verificar si hay algún segundo sin datos y mostrarlo
    missing_data = merged_df[merged_df.isnull().any(axis=1)]
    if not missing_data.empty:
        print("Timestamps sin datos disponibles:")
        print(missing_data[["DateTime"]])

    # Guardar el resultado en un archivo CSV
    merged_df.to_csv(output_file, index=False)
    print(f"Archivo combinado guardado en: {output_file}")
    return merged_df

def plot_data(merged_file):
    """Genera un gráfico con dos ejes: uno para los datos IMU y otro para las temperaturas."""
    
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Graficar AccelX en el eje primario (IMU)
    ax1.plot(merged_file["DateTime"], merged_file["AccelX"], label="AccelX", color='b', marker='.', linestyle='-', markersize=4)
    ax1.set_xlabel("Fecha/Hora")
    ax1.set_ylabel("AccelX", color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True)
    
    # Crear un segundo eje para las temperaturas
    ax2 = ax1.twinx()
    ax2.plot(merged_file["DateTime"], merged_file["Surface temperature (ºC)"], label="Surface Temp", color='g', marker='.', linestyle='-', markersize=4)
    ax2.plot(merged_file["DateTime"], merged_file["Over surface temperature (ºC)"], label="Over Surface Temp", color='orange', marker='.', linestyle='-', markersize=4)
    ax2.set_ylabel("Temperatura (ºC)", color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    
    # Formatear el eje X para mostrar horas y minutos
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.xticks(rotation=45)
    
    # Agregar leyendas separadas
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc='upper left')
    ax2.legend(lines2, labels2, loc='upper right')
    
    # Título del gráfico
    plt.title("Datos IMU y Temperatura - Datos del Sensor")
    
    # Ajustar el layout
    plt.tight_layout()
    
    # Mostrar el gráfico
    plt.show()

# Ejecutar la función
merged_data = merge_data(TEMP_FILE, IMU_FILE, OUTPUT_FILE)
plot_data(merged_data)
