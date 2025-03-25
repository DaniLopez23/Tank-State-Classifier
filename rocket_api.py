from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
import pandas as pd
import uvicorn
from sklearn.preprocessing import StandardScaler  # Añadir import

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

# Cargar modelo (sin scaler)
model_data = joblib.load(MODEL_PATH)
rocket = model_data['rocket']
classifier = model_data['classifier']
le = model_data['label_encoder']
window_config = model_data['window_config']
WINDOW_SIZE = window_config['window_size']
STEP_SIZE = window_config['step_size']  # Usar el paso del modelo

class PredictionRequest(BaseModel):
    AccelX: list[float]
    SurfaceTemperature: list[float]
    OverSurfaceTemperature: list[float]

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        windows, step = process_input(request)
        X_transformed = rocket.transform(windows)
        predictions = classifier.predict(X_transformed)

        final_predictions = []
        for i in range(len(request.AccelX)):
            votes = []
            for j, pred in enumerate(predictions):
                start = j * step
                end = start + WINDOW_SIZE
                if start <= i < end:
                    votes.append(pred)
            if votes:
                final_pred = le.inverse_transform([max(set(votes), key=votes.count)])[0]
            else:
                final_pred = "UNKNOWN"
            final_predictions.append(final_pred)

        final_predictions = ["UNKNOWN" if pd.isna(pred) else pred for pred in final_predictions]

        return {
            "predictions": final_predictions,
            "states": le.classes_.tolist()
        }
    
    except Exception as e:
        return {"error": str(e)}

def process_input(data: PredictionRequest):
    df = pd.DataFrame({
        'AccelX': data.AccelX,
        'Surface temperature (ºC)': data.SurfaceTemperature,
        'Over surface temperature (ºC)': data.OverSurfaceTemperature
    })
    
    df = df.fillna(0)

    if len(df) < WINDOW_SIZE:
        raise ValueError(f"Se requieren al menos {WINDOW_SIZE} muestras")

    step = STEP_SIZE
    raw_windows = [df.values[i:i+WINDOW_SIZE] for i in range(0, len(df) - WINDOW_SIZE + 1, step)]
    
    # Escalar cada ventana individualmente
    scaled_windows = []
    for window in raw_windows:
        scaler = StandardScaler()
        scaled_window = scaler.fit_transform(window)
        scaled_windows.append(scaled_window.T)  # Transponer para coincidir con la forma de Rocket
    
    return np.array(scaled_windows), step

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)