import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import roll_time_series
from sklearn.preprocessing import LabelEncoder

# --------------------------
# 1. Cargar y Fusionar Datos
# --------------------------
def load_and_merge_data(temp_path, imu_path):
    # Cargar datos
    df_temp = pd.read_csv(temp_path, sep=";", parse_dates=["Time (UTC)"])
    df_imu = pd.read_csv(imu_path, sep=";", parse_dates=["Time (UTC)"])

    # Renombrar columnas para claridad
    df_temp = df_temp.rename(columns={"Time (UTC)": "timestamp"})
    df_imu = df_imu.rename(columns={"Time (UTC)": "timestamp"})

    # Eliminar columna redundante y convertir timestamp
    df_temp = df_temp.drop("Submerged temperature (ÂºC)", axis=1)
    df_temp["timestamp"] = pd.to_datetime(df_temp["timestamp"])
    df_imu["timestamp"] = pd.to_datetime(df_imu["timestamp"])

    # Fusionar usando el timestamp más cercano
    df_merged = pd.merge_asof(
        df_imu.sort_values("timestamp"),
        df_temp.sort_values("timestamp"),
        on="timestamp",
        direction="nearest"
    )
    return df_merged

# --------------------------
# 2. Crear Ventanas Temporales
# --------------------------
def create_time_windows(df, window_minutes=10):
    # Crear ID de ventana cada X minutos
    df["ventana_id"] = df["timestamp"].astype("int64") // (1e9 * 60 * window_minutes)
    return df

# --------------------------
# 3. Extraer Características
# --------------------------
def extract_window_features(df):
    features = extract_features(
        df,
        column_id="ventana_id",
        column_sort="timestamp",
        default_fc_parameters={
            "mean": None,
            "standard_deviation": None,
            "maximum": None,
            "minimum": None,
            "linear_trend": [{"attr": "slope"}],
            "abs_energy": None
        },
        # Extraer características específicas por sensor:
        kind_to_fc_parameters={
            "Surface temperature (ÂºC)": {"linear_trend": [{"attr": "slope"}]},
            "Accel Y (G)": {"standard_deviation": None},
            "Gyro Z (rad/s)": {"maximum": None}
        }
    )
    return features.dropna(axis=1)  # Eliminar columnas con NaN

# --------------------------
# 4. Etiquetado Automático
# --------------------------
def label_data(features):
    # Reglas de etiquetado (¡Ajustar umbrales según dominio!)
    conditions = [
        (features["Surface temperature (ÂºC)__linear_trend__slope"] > 0.05) &
        (features["Accel Y (G)__standard_deviation"] > 0.1),
        
        (features["Surface temperature (ÂºC)__linear_trend__slope"].abs() < 0.02) &
        (features["Accel Y (G)__standard_deviation"] < 0.05)
    ]
    choices = ["MILKING", "MAINTENANCE"]
    
    features["estado"] = np.select(conditions, choices, default="DESCONOCIDO")
    return features

# --------------------------
# Ejecución Principal
# --------------------------
if __name__ == "__main__":
    # Parámetros configurables
    TEMP_PATH = "test_tank_temperature_probes_data.csv"
    IMU_PATH = "test_6_DoF_IMU_data.csv"
    WINDOW_MINUTES = 10  # Tamaño de ventana en minutos

    # Paso 1: Cargar y fusionar datos
    df_merged = load_and_merge_data(TEMP_PATH, IMU_PATH)
    
    # Paso 2: Crear ventanas temporales
    df_windowed = create_time_windows(df_merged, WINDOW_MINUTES)
    
    # Paso 3: Extraer características
    features = extract_window_features(df_windowed)
    
    # Paso 4: Etiquetar automáticamente
    labeled_data = label_data(features)
    
    # Guardar datos preprocesados
    labeled_data.to_csv("labeled_data.csv", index=False)
    print("Datos etiquetados guardados en 'labeled_data.csv'")

    # Opcional: Codificar etiquetas para modelos
    le = LabelEncoder()
    labeled_data["estado_encoded"] = le.fit_transform(labeled_data["estado"])
    print("\nEjemplo de etiquetas:", dict(zip(le.classes_, le.transform(le.classes_))))