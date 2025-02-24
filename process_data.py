# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

def cargar_y_fusionar(temp_path, imu_path):
    # Cargar y fusionar datos usando epoch timestamp
    df_temp = pd.read_csv(temp_path, sep=";")
    df_imu = pd.read_csv(imu_path, sep=";")

    # Convertir a enteros
    df_temp['Epoch timestamp (UTC)'] = df_temp['Epoch timestamp (UTC)'].astype(int)
    df_imu['Epoch timestamp (UTC)'] = df_imu['Epoch timestamp (UTC)'].astype(int)

    # Fusion exacta
    df = pd.merge(
        df_imu,
        df_temp,
        on='Epoch timestamp (UTC)',
        how='left',
        suffixes=('_IMU', '_TEMP')
    )
    
    # Crear timestamp a partir del epoch
    df['timestamp'] = pd.to_datetime(df['Epoch timestamp (UTC)'], unit='s', utc=True).dt.tz_convert(None)
    
    return df[['timestamp', 'Epoch timestamp (UTC)'] + 
             [c for c in df.columns if c not in ['timestamp', 'Epoch timestamp (UTC)', 'Source_IMU', 'Source_TEMP']]]

def crear_ventanas(df, window_minutes=10):
    # Crear ventanas temporales
    df['ventana_id'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() // (60 * window_minutes)
    return df

def graficar_datos(df):
    # Configurar gráfico interactivo
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Seleccionar columnas a graficar
    columnas_izq = ['Accel X (G)']  # Se mostrará en el eje principal (ax1)
    columnas_der = [
        'Surface temperature (ºC)',
        # 'Submerged temperature (ºC)',
        'Over surface temperature (ºC)'
    ]  # Se mostrará en el eje secundario (ax2)

    # Graficar las series del eje izquierdo
    for col in columnas_izq:
        if col in df.columns:
            ax1.plot(df['timestamp'], df[col], marker='o', linestyle='-', markersize=4, label=col, color='tab:blue')

    # Crear segundo eje Y
    ax2 = ax1.twinx()

    # Graficar las series del eje derecho
    for col in columnas_der:
        if col in df.columns:
            ax2.plot(df['timestamp'], df[col], marker='s', linestyle='-', markersize=4, label=col, alpha=0.7)

    # Personalizar ejes
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Accel X (G)', color='tab:blue')
    ax2.set_ylabel('Temperatura (ºC)', color='tab:red')

    # Configurar leyendas
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Personalizar gráfico
    plt.title('Series Temporales de Sensores')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Mostrar gráfico
    plt.show()

if __name__ == "__main__":
    # Procesamiento
    IMU_DATA = "6_DoF_IMU/6_DoF_IMU_data.30_09_2024.csv"
    TEMP_DATA = "tank_temperature_probes/tank_temperature_probes_data.30_09_2024.csv"
    
    datos = cargar_y_fusionar(
        "test_tank_temperature_probes_data.csv",
        "test_6_DoF_IMU_data.csv"
    )
    datos_con_ventanas = crear_ventanas(datos)
    
    # Guardar CSV
    datos_con_ventanas.to_csv("datos_fusionados.csv", index=False)
    
    # Generar gráfico interactivo
    graficar_datos(datos_con_ventanas)
    
    print("Proceso completado!")
    print("Archivo generado: datos_fusionados_final.csv")