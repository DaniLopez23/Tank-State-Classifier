import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Configuration
TEMPERATURE_THRESHOLDS = {
    'cleaning': 40,
    'inactive': (25, 35),
    'maintenance_surface': (3, 5),
    'maintenance_over': (5, 6.5)
}
PATH_FILE = 'merged_data.csv'
STATE_COLORS = {
    'CLEANING': 'red',
    'EMPTY': 'blue',
    'INACTIVE': 'gray',
    'MAINTENANCE': 'orange',
    'MILKING': 'green',
    'POST-MILKING': 'purple',
    'UNKNOWN': 'yellow'
}

def safe_nanmean(arr):
    """Calculate mean handling all-NaN cases."""
    with np.errstate(all='ignore'):
        return np.nanmean(arr) if not np.all(np.isnan(arr)) else np.nan

def safe_nanmax(arr):
    """Calculate max handling all-NaN cases."""
    with np.errstate(all='ignore'):
        return np.nanmax(arr) if not np.all(np.isnan(arr)) else np.nan

def calculate_trend(series):
    """Calculate temperature trend using linear regression."""
    temp_data = series.dropna()
    
    if len(temp_data) > 1:
        try:
            # Convert timestamp index to seconds since epoch
            times = temp_data.index.view('int64') // 10**9
            slope, _, _, _, _ = linregress(times, temp_data)
            return slope
        except Exception:
            return 0
    return 0

def classify_window(group):
    """Determine the classification state for a group (window) of data."""
    features = {
        'ventana_id': group['ventana_id'].iloc[0],
        'surface_temp_mean': safe_nanmean(group['Surface temperature (ºC)']),
        'over_temp_mean': safe_nanmean(group['Over surface temperature (ºC)']),
        'surface_temp_slope': calculate_trend(group['Surface temperature (ºC)']),
        'gyro_active': any(safe_nanmax(np.abs(group[col])) > 0.001 
                           for col in ['Gyro X (rad/s)', 'Gyro Y (rad/s)', 'Gyro Z (rad/s)']),
        'accel_x_mean': safe_nanmean(group['Accel X (G)'])
    }
    
    if safe_nanmax(group['Surface temperature (ºC)']) >= TEMPERATURE_THRESHOLDS['cleaning']:
        return 'CLEANING'
    if np.abs(features['accel_x_mean']) < 0.2 and not features['gyro_active']:
        return 'EMPTY'
    if (TEMPERATURE_THRESHOLDS['inactive'][0] <= features['surface_temp_mean'] <= TEMPERATURE_THRESHOLDS['inactive'][1] and
        TEMPERATURE_THRESHOLDS['inactive'][0] <= features['over_temp_mean'] <= TEMPERATURE_THRESHOLDS['inactive'][1] and
        not features['gyro_active']):
        return 'INACTIVE'
    if features['surface_temp_slope'] > 0.005 and features['gyro_active']:
        # Considerar también la temperatura de la superficie y la aceleración
        if features['surface_temp_mean'] < TEMPERATURE_THRESHOLDS['milking_max']:
            return 'MILKING'
    if features['surface_temp_slope'] < -0.005 and features['gyro_active']:
        return 'POST-MILKING'
    if (TEMPERATURE_THRESHOLDS['maintenance_surface'][0] <= features['surface_temp_mean'] <= TEMPERATURE_THRESHOLDS['maintenance_surface'][1] and
        TEMPERATURE_THRESHOLDS['maintenance_over'][0] <= features['over_temp_mean'] <= TEMPERATURE_THRESHOLDS['maintenance_over'][1] and
        features['gyro_active']):
        return 'MAINTENANCE'
    return 'UNKNOWN'

def load_and_preprocess_data():
    """Load CSV data, fill missing temperatures, and ensure timestamp is a column."""
    df = pd.read_csv(PATH_FILE, parse_dates=['timestamp'], na_values=[' ', '', 'NA'], keep_default_na=False)
    df.set_index('timestamp', inplace=True)
    df['Surface temperature (ºC)'] = df.groupby('ventana_id')['Surface temperature (ºC)'].ffill()
    df['Over surface temperature (ºC)'] = df.groupby('ventana_id')['Over surface temperature (ºC)'].ffill()
    df.reset_index(inplace=True)  # Now 'timestamp' is a column for plotting
    return df

def classify_rows(df):
    """Assign a classification to every row based on its ventana_id group."""
    df['state'] = np.nan
    for ventana_id, group in df.groupby('ventana_id'):
        state = classify_window(group)
        df.loc[group.index, 'state'] = state
    return df

def graficar_datos(df):
    """Genera un gráfico que muestra datos IMU y temperaturas en un mismo gráfico, 
    con las ventanas coloreadas según su clasificación."""
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Obtener fecha única para el título (suponiendo un solo día en los datos)
    fecha_unica = df['timestamp'].dt.date.unique()[0] if not df['timestamp'].empty else "Fecha desconocida"
    
    # Dibujar zonas de ventanas usando el color de la clasificación
    ventanas = df['ventana_id'].unique()
    for ventana in ventanas:
        ventana_df = df[df['ventana_id'] == ventana]
        if not ventana_df.empty:
            state = ventana_df['state'].iloc[0]
            color = STATE_COLORS.get(state, 'lightgray')
            ax1.axvspan(ventana_df['timestamp'].min(), ventana_df['timestamp'].max(),
                        color=color, alpha=0.2, label=f'Ventana {ventana} - {state}')
    
    # Graficar Gyro X en el eje primario
    ax1.plot(df['timestamp'], df['Gyro X (rad/s)'], 
             color='tab:blue', marker='.', linestyle='-', markersize=4, label='Gyro X (rad/s)')
    ax1.set_xlabel('Hora')
    ax1.set_ylabel('Gyro X (rad/s)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)
    
    # Configurar un segundo eje para temperaturas
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
    
    plt.title(f'Datos IMU y Temperatura - {fecha_unica}')
    plt.tight_layout()
    plt.show()

def main():
    df = load_and_preprocess_data()
    df = classify_rows(df)
    # Guardar los datos etiquetados si se requiere
    df.to_csv('labeled_tank_states.csv', index=False)
    graficar_datos(df)

if __name__ == '__main__':
    main()
