from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import pandas as pd
import uvicorn
from sklearn.preprocessing import StandardScaler
from typing import List

DATA_STRATEGY = "second"   # "second" o "minute"
MODEL_PATH = f"data_per_{DATA_STRATEGY}_strategy/model/trained_model.pkl"

app = FastAPI()

# Cargar modelo
model_data = joblib.load(MODEL_PATH)
rocket = model_data['rocket']
classifier = model_data['classifier']
le = model_data['label_encoder']
window_config = model_data['window_config']
WINDOW_SIZE = window_config['window_size']
STEP_SIZE = window_config['step_size']

class PredictionRequest(BaseModel):
    AccelX: List[float]
    SurfaceTemperature: List[float]
    OverSurfaceTemperature: List[float]

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # Validar longitud de datos
        if len(request.AccelX) != len(request.SurfaceTemperature) or len(request.AccelX) != len(request.OverSurfaceTemperature):
            raise HTTPException(status_code=400, detail="All input lists must have the same length")
        
        # Procesar entrada
        windows, timestamps = process_input(request)
        
        if len(windows) == 0:
            raise HTTPException(status_code=400, detail="Insufficient data to create windows")
        
        # Transformar características
        X_transformed = rocket.transform(windows)
        
        # Predecir
        predictions = classifier.predict(X_transformed)
        probabilities = classifier.predict_proba(X_transformed)
        
        # Convertir a etiquetas
        pred_labels = le.inverse_transform(predictions)
        confidence_scores = np.max(probabilities, axis=1).tolist()
        
        # Crear respuesta detallada
        response = {
            "predictions": pred_labels.tolist(),
            "confidence": confidence_scores,
            "windows": [
                {
                    "start": str(ts[0]),
                    "end": str(ts[1]),
                    "prediction": label,
                    "confidence": conf
                } for ts, label, conf in zip(timestamps, pred_labels, confidence_scores)
            ],
            "classes": le.classes_.tolist()
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def process_input(data: PredictionRequest):
    """Convierte los datos de entrada al formato requerido"""
    df = pd.DataFrame({
        'AccelX': data.AccelX,
        'Surface temperature (ºC)': data.SurfaceTemperature,
        'Over surface temperature (ºC)': data.OverSurfaceTemperature
    })
    
    # Manejar valores faltantes
    df = df.fillna(method='ffill').fillna(0)
    
    windows = []
    timestamps = []
    
    # Crear ventanas temporales
    for i in range(0, len(df) - WINDOW_SIZE + 1, STEP_SIZE):
        window_data = df.iloc[i:i+WINDOW_SIZE]
        
        # Escalado por ventana
        scaler = StandardScaler()
        scaled_window = scaler.fit_transform(window_data)
        
        # Almacenar timestamps
        start_time = i
        end_time = i + WINDOW_SIZE - 1
        timestamps.append((start_time, end_time))
        
        windows.append(scaled_window.T)  # Formato (features, window_size)
    
    return np.array(windows), timestamps

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)