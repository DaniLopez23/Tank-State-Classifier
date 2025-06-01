from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
import pandas as pd
import uvicorn
from datetime import datetime, timedelta
import logging
from typing import List
from sklearn.preprocessing import StandardScaler

DATA_STRATEGY = "5_second"
MODEL_PATH = f"data_per_{DATA_STRATEGY}_strategy/model/trained_model-new.pkl"

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
try:
    model_data = joblib.load(MODEL_PATH)
    rocket = model_data['rocket']
    pipeline = model_data['pipeline']
    le = model_data['label_encoder']
    window_config = model_data['window_config']
    WINDOW_SIZE = window_config['window_size']
    STEP_SIZE = window_config['step_size']
    logger.info(f"Model loaded successfully. Configuration: {window_config}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise RuntimeError("Failed to load model") from e

class SensorData(BaseModel):
    DateTime: str
    AccelX: float
    OverSurfaceTemperature: float
    SurfaceTemperature: float

class PredictionRequest(BaseModel):
    data: List[SensorData]

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # Crear DataFrame
        df = pd.DataFrame([{
            "DateTime": entry.DateTime,
            "AccelX": entry.AccelX,
            "Over surface temperature (ºC)": entry.OverSurfaceTemperature,
            "Surface temperature (ºC)": entry.SurfaceTemperature
        } for entry in request.data])
        
        # Validar datos
        if len(df) < WINDOW_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Se requieren al menos {WINDOW_SIZE} muestras"
            )

        # Procesar timestamps
        try:
            df['DateTime'] = pd.to_datetime(df['DateTime'], utc=True)
            timestamps = df['DateTime'].tolist()
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Formato DateTime inválido: {str(e)}"
            )

        # Crear ventanas
        windows = create_windows_api(df)
        
        # Transformar con Rocket
        try:
            X_rocket = rocket.transform(windows)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error en transformación Rocket: {str(e)}"
            )

        # Predecir con pipeline
        predictions = pipeline.predict(X_rocket)
        # Calcular decision_function solo si existe
        if hasattr(pipeline, "decision_function"):
            decision_scores = pipeline.decision_function(X_rocket)
            # Si es binario, decision_scores es 1D
            if len(decision_scores.shape) == 1:
                confidences = 1 / (1 + np.exp(-decision_scores)) * 100
            else:
                confidences = 1 / (1 + np.exp(-np.max(decision_scores, axis=1))) * 100
        else:
            confidences = [None] * len(predictions)

        # Generar intervalos
        intervals = []
        for j, (pred, conf) in enumerate(zip(predictions, confidences)):
            start_idx = j * STEP_SIZE
            end_idx = start_idx + WINDOW_SIZE
            if end_idx > len(timestamps):
                end_idx = len(timestamps)
            start_time = timestamps[start_idx]
            end_time = timestamps[end_idx - 1] if end_idx <= len(timestamps) else timestamps[-1]
            intervals.append({
                "inicio": start_time.strftime("%Y-%m-%d %H:%M:%S%z"),
                "fin": end_time.strftime("%Y-%m-%d %H:%M:%S%z"),
                "estado": le.inverse_transform([pred])[0],
                "confianza": round(float(conf), 1) if conf is not None else None
            })

        return {
            "intervals": intervals,
            "metrics": model_data['metrics'],
            "valid_classes": le.classes_.tolist()
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error interno del servidor: {str(e)}"
        )

def create_windows_api(df: pd.DataFrame):
    """Replica exactamente el preprocesamiento de entrenamiento"""
    features = df[["AccelX", "Surface temperature (ºC)", "Over surface temperature (ºC)"]].values
    n_samples = len(features)
    windows = []
    for i in range(0, n_samples, STEP_SIZE):
        end_idx = i + WINDOW_SIZE
        if end_idx > n_samples:
            padding = end_idx - n_samples
            window = np.pad(features[i:n_samples], ((0, padding), (0, 0)), mode='edge')
        else:
            window = features[i:end_idx]
        scaler = StandardScaler()
        window = scaler.fit_transform(window)
        windows.append(window.T)  # (features, timesteps)
    return np.array(windows)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)