import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sktime.transformations.panel.rocket import Rocket
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay, f1_score
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import TimeSeriesSplit

# Configuración
DATA_STRATEGY = "5_second"   # "second" o "minute" o "5_second"

if DATA_STRATEGY == "second":
    WINDOW_SIZE = 1800
    STEP_SIZE = 900
elif DATA_STRATEGY == "minute":
    WINDOW_SIZE = 30
    STEP_SIZE = 15
else:
    WINDOW_SIZE = 360
    STEP_SIZE = 180
    
DATA_TRAIN_DIR = f"data_per_{DATA_STRATEGY}_strategy/data/train"
DATA_TEST_DIR = f"data_per_{DATA_STRATEGY}_strategy/data/test"
DATA_VALID_DIR = f"data_per_{DATA_STRATEGY}_strategy/data/valid"

MODEL_SAVE_PATH = f"data_per_{DATA_STRATEGY}_strategy/model/trained_model.pkl"
REPORT_DIR = f"data_per_{DATA_STRATEGY}_strategy/reports"

# Configuración de Rocket
ROCKET_KERNELS = {
    "second": 10000,
    "minute": 10000,
    "5_second": 5000
}

# Configuración de LightGBM
LGBM_PARAMS = {
    "second": {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "n_estimators": 1000,
        "learning_rate": 0.1,
        "num_leaves": 128,
        "max_depth": 5,
        "min_data_in_leaf": 20,
        "colsample_bytree": 0.2,  # Reemplaza feature_fraction
        "subsample": 0.8,         # Reemplaza bagging_fraction
        "subsample_freq": 10,      # Reemplaza bagging_freq
        "reg_alpha": 0.05,         # En vez de lambda_l1
        "reg_lambda": 0.05,        # En vez de lambda_l2
        "force_col_wise": True,
        "random_state": 42
    },
    "minute": {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "n_estimators": 200,
        "learning_rate": 0.2,
        "num_leaves": 64,
        "max_depth": 7,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "reg_alpha": 0.05,  
        "reg_lambda": 0.05,
        "n_jobs": -1,
        "random_state": 42
    },
    "5_second": {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "n_estimators": 600,
        "learning_rate": 0.05,
        "num_leaves": 128,
        "max_depth": 5,
        "min_data_in_leaf": 20,
        "colsample_bytree": 0.2,  # Reemplaza feature_fraction
        "subsample": 0.8,         # Reemplaza bagging_fraction
        "subsample_freq": 10,      # Reemplaza bagging_freq
        "reg_alpha": 0.05,         # En vez de lambda_l1
        "reg_lambda": 0.05,        # En vez de lambda_l2
        "force_col_wise": True,
        "random_state": 42
    }
}

NUM_KERNELS = ROCKET_KERNELS[DATA_STRATEGY]
LGBM_CONFIG = LGBM_PARAMS[DATA_STRATEGY]

def load_and_preprocess_data(folder_path):
    """Carga y preprocesa todos los CSVs en un directorio"""
    all_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
    dfs = []
    
    for file in all_files:
        print(f"Processing {file}...")
        df = pd.read_csv(os.path.join(folder_path, file), parse_dates=["DateTime"])
        
        # Verificar y manejar valores faltantes
        if df.isnull().any().any():
            print(f"Valores faltantes en {file}:")
            print(df.isnull().sum())
            df.ffill(inplace=True)  # Forward fill para series temporales
        
        dfs.append(df)
    
    full_df = pd.concat(dfs, ignore_index=True)

    features = full_df[["AccelX", "Surface temperature (ºC)", "Over surface temperature (ºC)"]]
    labels = full_df["ETIQUETA"]

    print(f"\nTotal de muestras: {len(full_df)}")
    print("Distribución de clases:")
    print(labels.value_counts(normalize=True).apply(lambda x: f"{x:.2%}"), "\n")

    return features.values, labels.values, full_df

def create_windows(features, labels, window_size, step):
    """Crea ventanas temporales con escalado independiente para cada ventana"""
    X, y = [], []
    for i in range(0, len(features) - window_size + 1, step):
        # Escalado por ventana para evitar data leakage
        scaler = StandardScaler()
        X_window = scaler.fit_transform(features[i:i+window_size])
        y_window = labels[i:i+window_size]
        
        # Determinar etiqueta (según lógica actual)
        label = max(set(y_window), key=list(y_window).count)
        
        X.append(X_window.T)  # Formato (n_variables, window_size)
        y.append(label)
    
    print(f"\nVentanas creadas: {len(X)}")
    print(f"Tamaño de cada ventana: {window_size} muestras")
    print(f"Paso entre ventanas: {step} muestras\n")
    return np.array(X), np.array(y)

def apply_smote(X, y, window_size):
    """Aplica SMOTE en el espacio de características de Rocket"""
    print("\nAplicando SMOTE en el espacio de características...")
    smote = SMOTE(random_state=42, k_neighbors=min(5, Counter(y)[1]-1))
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res

def evaluate_model(model, X, y, label_encoder, split_type="Train"):
    """Evaluación avanzada con múltricas adicionales"""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    
    print(f"\n{split_type} Classification Report:")
    print(classification_report(y, y_pred, target_names=label_encoder.classes_))
    
    # Métricas adicionales
    f1 = f1_score(y, y_pred, average='weighted')
    print(f"Weighted F1-Score: {f1:.3f}")
    
    # Matriz de confusión
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plt.title(f"{split_type} Confusion Matrix")
    plt.savefig(f"{REPORT_DIR}/{split_type.lower()}_confusion_matrix.png")
    plt.close()
    
    return accuracy_score(y, y_pred), f1

def train_model(X_train, y_train, X_valid, y_valid):
    """Entrenamiento con validación temprana"""
    model = LGBMClassifier(**LGBM_CONFIG)
    
    model.fit(
        X_train, 
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="multi_logloss",
        callbacks=[early_stopping(stopping_rounds=50, verbose=1)]
    )
    
    return model

# Crear directorios necesarios
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# Cargar y procesar datos
print("\n" + "="*50)
print("Cargando datos de entrenamiento...")
X_train_raw, y_train_raw, df_train = load_and_preprocess_data(DATA_TRAIN_DIR)

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train_raw)

# Crear ventanas temporales
print("Creando ventanas de entrenamiento...")
X_train_windows, y_train_windows = create_windows(X_train_raw, y_train_encoded, WINDOW_SIZE, STEP_SIZE)

# Procesar datos de validación
print("\n" + "="*50)
print("Cargando datos de validación...")
X_valid_raw, y_valid_raw, _ = load_and_preprocess_data(DATA_VALID_DIR)
y_valid_encoded = le.transform(y_valid_raw)
X_valid_windows, y_valid_windows = create_windows(X_valid_raw, y_valid_encoded, WINDOW_SIZE, STEP_SIZE)

# Entrenar modelo Rocket
print("\n" + "="*50)
print("Entrenando transformación Rocket...")
rocket = Rocket(num_kernels=NUM_KERNELS, random_state=42)
rocket.fit(X_train_windows)

# Transformar datos
X_train_transformed = rocket.transform(X_train_windows)
X_valid_transformed = rocket.transform(X_valid_windows)

# Balancear clases con SMOTE
X_train_resampled, y_train_resampled = apply_smote(X_train_transformed, y_train_windows, WINDOW_SIZE)

# Entrenar modelo final
print("\n" + "="*50)
print("Entrenando clasificador LightGBM...")
classifier = train_model(X_train_resampled, y_train_resampled, X_valid_transformed, y_valid_windows)

# Evaluación completa
print("\n" + "="*50)
train_acc, train_f1 = evaluate_model(classifier, X_train_resampled, y_train_resampled, le, "Train")
valid_acc, valid_f1 = evaluate_model(classifier, X_valid_transformed, y_valid_windows, le, "Validation")

# Evaluación en test si existe
test_acc, test_f1 = None, None
if os.path.exists(DATA_TEST_DIR) and os.listdir(DATA_TEST_DIR):
    try:
        print("\n" + "="*50)
        print("Evaluando en conjunto de test...")
        X_test_raw, y_test_raw, _ = load_and_preprocess_data(DATA_TEST_DIR)
        y_test_encoded = le.transform(y_test_raw)
        X_test_windows, y_test_windows = create_windows(X_test_raw, y_test_encoded, WINDOW_SIZE, STEP_SIZE)
        X_test_transformed = rocket.transform(X_test_windows)
        test_acc, test_f1 = evaluate_model(classifier, X_test_transformed, y_test_windows, le, "Test")
    except Exception as e:
        print(f"Error en evaluación de test: {str(e)}")

# Guardar modelo y reportes
joblib.dump({
    'rocket': rocket,
    'classifier': classifier,
    'label_encoder': le,
    'window_config': {
        'window_size': WINDOW_SIZE,
        'step_size': STEP_SIZE
    },
    'metrics': {
        'train': {'accuracy': train_acc, 'f1': train_f1},
        'validation': {'accuracy': valid_acc, 'f1': valid_f1},
        'test': {'accuracy': test_acc, 'f1': test_f1} if test_acc else None
    }
}, MODEL_SAVE_PATH)

print("\n" + "="*50)
print("Entrenamiento completado exitosamente!")
print(f"Modelo guardado en: {MODEL_SAVE_PATH}")