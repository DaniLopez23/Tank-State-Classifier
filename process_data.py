# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

def cargar_y_fusionar(imu_path, temp_path):
    # Load IMU and Temperature data
    df_imu = pd.read_csv(imu_path, sep=";")
    df_temp = pd.read_csv(temp_path, sep=";")

    # Ensure timestamp is integer
    df_imu['Epoch timestamp (UTC)'] = df_imu['Epoch timestamp (UTC)'].astype(int)
    df_temp['Epoch timestamp (UTC)'] = df_temp['Epoch timestamp (UTC)'].astype(int)

    # Merge using only 'Epoch timestamp (UTC)' (LEFT JOIN to keep IMU data)
    df = pd.merge(
        df_imu,
        df_temp[['Epoch timestamp (UTC)', 'Surface temperature (ºC)', 'Over surface temperature (ºC)']],
        on='Epoch timestamp (UTC)',
        how='left'  # Keep all IMU rows, merge available temperature data
    )

    # Create readable timestamp column
    df['timestamp'] = pd.to_datetime(df['Epoch timestamp (UTC)'], unit='s', utc=True).dt.tz_convert(None)

    # Reorder columns
    return df[['timestamp', 'Epoch timestamp (UTC)'] + 
              [c for c in df.columns if c not in ['timestamp', 'Epoch timestamp (UTC)', 'Source_IMU', 'Source_TEMP']]]

def crear_ventanas(df, window_minutes=10):
    # Create rolling windows for analysis
    df['ventana_id'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() // (60 * window_minutes)
    return df

def graficar_datos(df):
    # Configure interactive plot
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Select columns to plot
    columnas_izq = ['Accel X (G)']  # Primary Y-axis (left)
    columnas_der = ['Submerged temperature (ºC)', 'Surface temperature (ºC)', 'Over surface temperature (ºC)']  # Secondary Y-axis (right)

    # Plot left Y-axis data
    for col in columnas_izq:
        if col in df.columns:
            ax1.plot(df['timestamp'], df[col], marker='o', linestyle='-', markersize=4, label=col, color='tab:blue')

    # Create secondary Y-axis
    ax2 = ax1.twinx()

    # Plot right Y-axis data
    for col in columnas_der:
        if col in df.columns:
            ax2.plot(df['timestamp'], df[col], marker='s', linestyle='-', markersize=4, label=col, alpha=0.7)

    # Customize axes
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Accel X (G)', color='tab:blue')
    ax2.set_ylabel('Temperature (ºC)', color='tab:red')

    # Set legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Final plot settings
    plt.title('Sensor Time Series Data')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Show plot
    plt.show()

if __name__ == "__main__":
    # Configuración de rutas
    IMU_DATA = "6_DoF_IMU/6_DoF_IMU_data.23_08_2024.csv"
    TEMP_DATA = "tank_temperature_probes/tank_temperature_probes_data.23_08_2024.csv"

    # Show sample input data
    print("Datos IMU:")
    print(pd.read_csv(IMU_DATA, sep=";").head())
    print("\nDatos de Temperatura:")
    print(pd.read_csv(TEMP_DATA, sep=";").head())

    # Process data
    datos = cargar_y_fusionar(IMU_DATA, TEMP_DATA)
    datos_con_ventanas = crear_ventanas(datos)

    # Save to CSV
    datos_con_ventanas.to_csv("datos_fusionados.csv", index=False)

    # Generate plot
    graficar_datos(datos_con_ventanas)

    print("Process completed!")
    print("Generated file: datos_fusionados.csv")
