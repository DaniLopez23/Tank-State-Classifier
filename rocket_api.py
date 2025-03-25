from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
import pandas as pd
import uvicorn
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import logging

DATA_STRATEGY = "5_second"
MODEL_PATH = f"data_per_{DATA_STRATEGY}_strategy/model/trained_model.pkl"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("uvicorn")

# Cargar modelo
model_data = joblib.load(MODEL_PATH)
rocket = model_data['rocket']
classifier = model_data['classifier']
le = model_data['label_encoder']
window_config = model_data['window_config']
WINDOW_SIZE = window_config['window_size']

logger.info(f"Model configuration: {window_config}")

class SensorData(BaseModel):
    DateTime: str
    AccelX: float
    OverSurfaceTemperature: float
    SurfaceTemperature: float

class PredictionRequest(BaseModel):
    data: list[SensorData]

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # Crear DataFrame desde los datos de entrada
        df = pd.DataFrame([{
            "DateTime": entry.DateTime,
            "AccelX": entry.AccelX,
            "Over surface temperature (ºC)": entry.OverSurfaceTemperature,
            "Surface temperature (ºC)": entry.SurfaceTemperature
        } for entry in request.data])

        if len(df) < WINDOW_SIZE:
            raise ValueError(f"Se requieren al menos {WINDOW_SIZE} muestras")

        # Procesar timestamps
        try:
            timestamps = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S%z") for ts in df["DateTime"]]
        except Exception as e:
            raise ValueError(f"Formato DateTime inválido: {str(e)}")

        # Calcular duración de muestra
        sample_duration = (timestamps[1] - timestamps[0]).total_seconds() if len(timestamps) > 1 else 5

        # Procesar datos
        windows = process_input(df)
        X_transformed = rocket.transform(windows)
        predictions = classifier.predict(X_transformed)

        # Generar intervalos por ventana
        intervals = []
        for j in range(len(predictions)):
            start_idx = j * WINDOW_SIZE
            end_idx = start_idx + WINDOW_SIZE - 1
            
            if end_idx >= len(timestamps):
                end_idx = len(timestamps) - 1

            start_time = timestamps[start_idx]
            end_time = timestamps[end_idx] + timedelta(seconds=sample_duration)
            
            intervals.append({
                "inicio": start_time.strftime("%d/%m/%Y %H:%M:%S"),
                "fin": end_time.strftime("%d/%m/%Y %H:%M:%S"),
                "estado": le.inverse_transform([predictions[j]])[0]
            })

        return {
            "intervals": intervals,
            "states": le.classes_.tolist()
        }

    except Exception as e:
        return {"error": str(e)}

def process_input(data: pd.DataFrame):
    df = data.drop(columns=["DateTime"]).fillna(0)
    
    if len(df) < WINDOW_SIZE:
        raise ValueError(f"Se requieren al menos {WINDOW_SIZE} muestras")

    # Crear ventanas completas sin overlap
    n_windows = len(df) // WINDOW_SIZE
    valid_length = n_windows * WINDOW_SIZE
    df = df.iloc[:valid_length]
    
    raw_windows = [df.values[i:i+WINDOW_SIZE] for i in range(0, valid_length, WINDOW_SIZE)]
    
    # Escalar cada ventana
    scaled_windows = []
    for window in raw_windows:
        scaler = StandardScaler()
        scaled_window = scaler.fit_transform(window)
        scaled_windows.append(scaled_window.T)
    
    return np.array(scaled_windows)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)