import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

WINDOW_MINUTES = 120  # Duración de cada ventana en minutos


def cargar_datos(path, required_columns):
    """Carga un archivo CSV y filtra filas con datos faltantes"""
    try:
        df = pd.read_csv(path, sep=";")
        
        # Eliminar filas con valores nulos en las columnas requeridas
        df_valid = df.dropna(subset=required_columns)
        df_invalid = df[~df.index.isin(df_valid.index)]

        # Imprimir las líneas descartadas
        if not df_invalid.empty:
            print(f"Líneas descartadas en {path} por datos incompletos:")
            print(df_invalid.to_string(index=False), "\n")

        return df_valid

    except Exception as e:
        print(f"Error al cargar {path}: {e}")
        return pd.DataFrame()


def fusionar_datos(df_imu, df_temp):
    """Fusiona los DataFrames de IMU y temperatura en un solo conjunto de datos."""
    df_imu = df_imu.drop(columns=['Source', 'Date', 'Time (UTC)'])
    df_temp = df_temp.drop(columns=['Source', 'Date', 'Time (UTC)'])

    # Convertir timestamps a enteros para evitar errores de fusión
    df_imu['Epoch timestamp (UTC)'] = df_imu['Epoch timestamp (UTC)'].astype(int)
    df_temp['Epoch timestamp (UTC)'] = df_temp['Epoch timestamp (UTC)'].astype(int)

    # Fusionar los DataFrames por timestamp
    df = pd.merge(
        df_imu,
        df_temp[['Epoch timestamp (UTC)', 'Surface temperature (ºC)', 'Over surface temperature (ºC)']],
        on='Epoch timestamp (UTC)',
        how='outer'
    )

    # Convertir el timestamp a datetime sin zona horaria
    df['timestamp'] = pd.to_datetime(df['Epoch timestamp (UTC)'], unit='s', utc=True).dt.tz_convert(None)

    return df[['timestamp', 'Epoch timestamp (UTC)'] + [c for c in df.columns if c not in ['timestamp', 'Epoch timestamp (UTC)']]]


def crear_ventanas(df):
    """Crea ventanas de tiempo y asigna un ID de ventana a cada dato."""
    df['ventana_id'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() // (60 * WINDOW_MINUTES)
    return df


def graficar_datos(df):
    """Genera un gráfico de los datos de IMU y temperatura, mostrando las ventanas correctamente."""
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Obtener fecha única para el título
    fecha_unica = df['timestamp'].dt.date.unique()[0]

    # Dibujar zonas de ventanas alternando colores
    ventanas = df['ventana_id'].unique()
    for i, ventana in enumerate(ventanas):
        ventana_df = df[df['ventana_id'] == ventana]
        if not ventana_df.empty:
            color = 'gray' if i % 2 == 0 else 'lightgray'
            ax1.axvspan(ventana_df['timestamp'].min(), ventana_df['timestamp'].max(),
                        color=color, alpha=0.2)

    # Graficar Gyro X
    ax1.plot(df['timestamp'], df['Gyro X (rad/s)'], 
             color='tab:blue', marker='.', linestyle='-', markersize=4, label='Aceleración X')

    ax1.set_xlabel('Hora')
    ax1.set_ylabel('Gyro X (rad/s)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)

    # Configurar eje derecho para temperaturas
    ax2 = ax1.twinx()
    temp_columns = ['Surface temperature (ºC)', 'Over surface temperature (ºC)']
    colors = ['tab:red', 'tab:green']

    for col, color in zip(temp_columns, colors):
        if col in df.columns:
            temp_data = df[['timestamp', col]].dropna()
            ax2.plot(temp_data['timestamp'], temp_data[col], 
                     color=color, marker='.', linestyle='-', markersize=4, linewidth=1.5, alpha=0.8, label=col)

    ax2.set_ylabel('Temperatura (ºC)')
    ax2.tick_params(axis='y')

    # Formatear el eje X para mostrar horas y minutos
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.xticks(rotation=45)

    # Agregar leyendas combinadas
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Agregar título con la fecha
    plt.title(f'Datos IMU y Temperatura - {fecha_unica}')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    IMU_DATA = "6_DoF_IMU/6_DoF_IMU_data.25_08_2024.csv"
    TEMP_DATA = "tank_temperature_probes/tank_temperature_probes_data.25_08_2024.csv"

    # Columnas requeridas en cada archivo
    required_columns_imu = ['Epoch timestamp (UTC)', 'Gyro X (rad/s)']
    required_columns_temp = ['Epoch timestamp (UTC)', 'Surface temperature (ºC)', 'Over surface temperature (ºC)']

    # Cargar datos con validación
    df_imu = cargar_datos(IMU_DATA, required_columns_imu)
    df_temp = cargar_datos(TEMP_DATA, required_columns_temp)

    # Fusionar datos
    if not df_imu.empty and not df_temp.empty:
        datos = fusionar_datos(df_imu, df_temp)
        datos_ventanas = crear_ventanas(datos)

        # Guardar el CSV filtrado
        datos_ventanas.to_csv("merged_data.csv", index=False)

        # Graficar
        graficar_datos(datos_ventanas)

        print("Proceso completado! Ver:")
        print("- datos_fusionados.csv")
        print("- Gráfico interactivo")
    else:
        print("Error: No se pudo generar el gráfico debido a datos insuficientes.")
