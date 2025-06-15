import optuna
from sktime.transformations.panel.rocket import Rocket
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sktime.transformations.panel.rocket import Rocket
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from collections import Counter

# Configuración
DATA_STRATEGY = "second"

WINDOW_SIZE = 1200  # 30 minutos en segundos
DATA_TRAIN_DIR = f"data_per_{DATA_STRATEGY}_strategy/data/train"
DATA_TEST_DIR = f"data_per_{DATA_STRATEGY}_strategy/data/test"
DATA_VALID_DIR = f"data_per_{DATA_STRATEGY}_strategy/data/valid"

MODEL_SAVE_PATH = "model/rocket/trained_model.pkl"
REPORT_DIR = "reports/rocket_reports"

def load_and_preprocess_data(folder_path):
    """Carga y preprocesa todos los CSVs en un directorio"""
    all_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
    dfs = []
    
    for file in all_files:
        print(f"Processing {file}...")
        df = pd.read_csv(os.path.join(folder_path, file), parse_dates=["DateTime"])

        # Verificar valores vacíos (strings vacíos)
        empty_values = (df == "").sum()
        if empty_values.any():
            print(f"Valores vacíos en {file}:\n{empty_values[empty_values > 0]}\n")

        dfs.append(df)
    
    full_df = pd.concat(dfs, ignore_index=True)

    features = full_df[["AccelX", "Surface temperature (ºC)", "Over surface temperature (ºC)"]]
    labels = full_df["ETIQUETA"]

    print(f"Total de muestras: {len(full_df)}\n")
    print(f"Clases: {labels.value_counts()}\n")

    return features.values, labels.values, full_df

# Función para optimizar Rocket + LightGBM
def objective(trial):
    # Hiperparámetros de Rocket
    num_kernels = trial.suggest_int("num_kernels", 3000, 10000, step=1000)

    # Hiperparámetros de LightGBM
    n_estimators = trial.suggest_int("n_estimators", 100, 500, step=50)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
    num_leaves = trial.suggest_int("num_leaves", 20, 100, step=10)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    subsample = trial.suggest_float("subsample", 0.6, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
    reg_alpha = trial.suggest_float("reg_alpha", 0.0, 1.0)
    reg_lambda = trial.suggest_float("reg_lambda", 0.0, 1.0)

    # Transformación con Rocket
    rocket = Rocket(num_kernels=num_kernels, random_state=42)
    rocket.fit(X_train_resampled)
    X_train_transformed = rocket.transform(X_train_resampled)
    X_valid_transformed = rocket.transform(X_valid_windows)

    # Entrenamiento de LightGBM
    classifier = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=42
    )
    classifier.fit(X_train_transformed, y_train_resampled)

    # Evaluación del modelo
    y_pred = classifier.predict(X_valid_transformed)
    accuracy = accuracy_score(y_valid, y_pred)

    return accuracy  # Optuna maximiza la métrica (accuracy)

# Ejecutar la optimización
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)  # Ejecuta 30 intentos de búsqueda

# Mostrar mejores hiperparámetros
print("Mejores parámetros encontrados:", study.best_params)
