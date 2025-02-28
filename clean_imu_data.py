import pandas as pd
import os

def split_csv_by_date(input_csv, output_folder, timestamp_column='_time'):
    # Crear la carpeta de salida si no existe
    os.makedirs(output_folder, exist_ok=True)
    
    # Cargar el CSV en un DataFrame, omitiendo las líneas de metadatos de InfluxDB
    df = pd.read_csv(input_csv, skiprows=3)
    
    # Asegurar que la columna de tiempo esté en formato datetime
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    
    # Filtrar solo las columnas necesarias
    df = df[[timestamp_column, '_value', '_field']]
    
    # Filtrar solo los datos de AccelX
    df = df[df['_field'] == 'fields_accel_x']
    
    # Renombrar la columna para claridad
    df = df.rename(columns={'_value': 'AccelX'})
    
    # Extraer la fecha (sin hora) para agrupar los datos
    df['date'] = df[timestamp_column].dt.date
    
    # Iterar sobre los grupos por fecha y guardar cada uno en un CSV separado
    for date, group in df.groupby('date'):
        output_file = os.path.join(output_folder, f"accel_data_{date}.csv")
        group.drop(columns=['date', '_field'], inplace=True)  # Eliminar columnas innecesarias
        group.to_csv(output_file, index=False)
        print(f"Archivo guardado: {output_file}")

# Uso del script
input_csv = 'accel.csv'  # Reemplázalo con tu archivo CSV
output_folder = 'accel_data_by_date'  # Carpeta donde se guardarán los archivos
split_csv_by_date(input_csv, output_folder)
