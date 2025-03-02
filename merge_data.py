import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DATE = "2024-08-27"

# Definir rutas de los archivos
TEMP_FILE = f"temp_data/temp_data_{DATE}.csv"
IMU_FILE = f"accel_data/accel_data_{DATE}.csv"

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
    
    # Fusionar los datos por timestamp, manteniendo todos los valores de IMU
    merged_df = pd.merge_asof(imu_df.sort_values("DateTime"),
                              temp_df.sort_values("DateTime"),
                              on="DateTime",
                              direction="backward")
    
    # Reemplazar el primer valor de temperatura si es NaN usando 'loc' para evitar la advertencia de copia
    if pd.isna(merged_df.loc[0, "Surface temperature (ºC)"]):  # Si la primera temperatura es NaN
        merged_df.loc[0, "Surface temperature (ºC)"] = STANDARD_INIT_SURFACE_TEMP
    if pd.isna(merged_df.loc[0, "Over surface temperature (ºC)"]):  # Si la primera temperatura sobre la superficie es NaN
        merged_df.loc[0, "Over surface temperature (ºC)"] = STANDARD_INIT_OVER_SURFACE_TEMP
    if pd.isna(merged_df.loc[0, "Submerged temperature (ºC)"]):  # Si la primera temperatura sumergida es NaN
        merged_df.loc[0, "Submerged temperature (ºC)"] = STANDARD_INIT_SUBMERGED_TEMP
    
    # Lógica para los valores siguientes: Si no existen, tomar el valor más cercano
    for col in ["Surface temperature (ºC)", "Over surface temperature (ºC)", "Submerged temperature (ºC)"]:
        merged_df[col] = merged_df[col].bfill()  # Usar 'bfill()' directamente para rellenar con el valor siguiente más cercano
    
    # Eliminar las columnas '_time_x' y '_time_y' generadas por el merge
    merged_df = merged_df.drop(columns=["_time_x", "_time_y"], errors="ignore")
    
    merged_df = merged_df.drop(columns="Submerged temperature (ºC)", errors="ignore")
    
    # Guardar el resultado en un archivo CSV
    merged_df.to_csv(output_file, index=False)
    print(f"Archivo combinado guardado en: {output_file}")
    return merged_df


def plot_data(merged_file):
    """Genera un gráfico con dos ejes: uno para los datos IMU y otro para las temperaturas,
    con las leyendas separadas para cada tipo de dato."""
    
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
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # Formato para mostrar horas y minutos
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))  # Mostrar una marca cada hora
    plt.xticks(rotation=45)
    
    # Agregar leyendas separadas
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    # Leyenda para el eje IMU
    ax1.legend(lines1, labels1, loc='upper left')
    
    # Leyenda para el eje de temperatura
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
