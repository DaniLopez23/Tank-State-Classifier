# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

def cargar_y_fusionar(imu_path, temp_path):
    # Cargar datos IMU y Temperatura
    df_imu = pd.read_csv(imu_path, sep=";")
    df_temp = pd.read_csv(temp_path, sep=";")

    # Convertir timestamps a enteros
    df_imu['Epoch timestamp (UTC)'] = df_imu['Epoch timestamp (UTC)'].astype(int)
    df_temp['Epoch timestamp (UTC)'] = df_temp['Epoch timestamp (UTC)'].astype(int)

    # Fusión LEFT JOIN usando solo el epoch timestamp
    df = pd.merge(
        df_imu,
        df_temp[['Epoch timestamp (UTC)', 'Surface temperature (ºC)', 'Over surface temperature (ºC)']],
        on='Epoch timestamp (UTC)',
        how='left'
    )

    # Crear timestamp legible
    df['timestamp'] = pd.to_datetime(df['Epoch timestamp (UTC)'], unit='s', utc=True).dt.tz_convert(None)
    
    return df[['timestamp', 'Epoch timestamp (UTC)'] + 
            [c for c in df.columns if c not in ['timestamp', 'Epoch timestamp (UTC)', 'Source']]]

def crear_ventanas(df, window_minutes=10):
    # Crear ventanas temporales
    df['ventana_id'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() // (60 * window_minutes)
    return df

def graficar_datos(df):
    # Configurar figura y ejes
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Graficar acelerómetro en eje izquierdo
    ax1.plot(df['timestamp'], df['Accel X (G)'], 
            color='tab:blue', 
            marker='o',
            linestyle='-',
            markersize=4,
            label='Aceleración X')
    
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Aceleración (G)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)

    # Configurar eje derecho para temperaturas
    ax2 = ax1.twinx()
    
    # Graficar temperaturas con líneas continuas
    temp_columns = ['Surface temperature (ºC)', 'Over surface temperature (ºC)']
    colors = ['tab:red', 'tab:green']
    
    for col, color in zip(temp_columns, colors):
        if col in df.columns:
            # Filtrar NaN y mantener orden temporal
            temp_data = df[['timestamp', col]].dropna()
            
            ax2.plot(temp_data['timestamp'], 
                    temp_data[col], 
                    color=color,
                    marker='o',
                    linestyle='-',
                    markersize=4,
                    linewidth=1.5,
                    alpha=0.8,
                    label=col)

    ax2.set_ylabel('Temperatura (ºC)')
    ax2.tick_params(axis='y')
    
    # Leyendas combinadas
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title('Datos IMU y Temperatura Fusionados')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Rutas de archivos
    IMU_DATA = "6_DoF_IMU/6_DoF_IMU_data.23_08_2024.csv"
    TEMP_DATA = "tank_temperature_probes/tank_temperature_probes_data.23_08_2024.csv"

    # Procesar datos
    datos = cargar_y_fusionar(IMU_DATA, TEMP_DATA)
    datos_ventanas = crear_ventanas(datos)
    
    # Guardar y mostrar gráfico
    datos_ventanas.to_csv("datos_fusionados.csv", index=False)
    graficar_datos(datos_ventanas)
    
    print("Proceso completado! Ver:")
    print("- datos_fusionados.csv")
    print("- Gráfico interactivo")