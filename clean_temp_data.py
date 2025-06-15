import pandas as pd
import os

def split_csv_by_date(input_csv, output_folder, timestamp_column='_time'):
    # Crear la carpeta de salida si no existe
    os.makedirs(output_folder, exist_ok=True)
    
    # Cargar el CSV en un DataFrame, omitiendo las líneas de metadatos de InfluxDB
    df = pd.read_csv(input_csv, skiprows=3)
    
    # Asegurar que la columna de tiempo esté en formato datetime
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    
    # Seleccionar solo las columnas necesarias y pivotar los datos
    df = df[['table', timestamp_column, '_value', '_field']]
    df_pivot = df.pivot(index=timestamp_column, columns='_field', values='_value')
    
    # Renombrar columnas para el formato requerido
    df_pivot = df_pivot.rename(columns={
        'fields_surface_temperature': 'Surface temperature (ºC)',
        'fields_over_surface_temperature': 'Over surface temperature (ºC)',
        'fields_submerged_temperature': 'Submerged temperature (ºC)'
    })
    
    # Resetear el índice para que _time vuelva a ser una columna normal
    df_pivot.reset_index(inplace=True)
    
    # Extraer la fecha (sin hora) para agrupar los datos
    df_pivot['date'] = df_pivot[timestamp_column].dt.date
    
    # Iterar sobre los grupos por fecha y guardar cada uno en un CSV separado
    for date, group in df_pivot.groupby('date'):
        output_file = os.path.join(output_folder, f"temp_data_{date}.csv")
        group.drop(columns=['date'], inplace=True)  # Eliminar la columna auxiliar de fecha
        group.to_csv(output_file, index=False)
        print(f"Archivo guardado: {output_file}")

# Uso del script
input_csv = 'temp.csv'  # Reemplázalo con tu archivo CSV
output_folder = 'temp_data'  # Carpeta donde se guardarán los archivos
split_csv_by_date(input_csv, output_folder)
