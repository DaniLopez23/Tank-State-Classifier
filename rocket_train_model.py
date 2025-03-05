import os
import pandas as pd
import numpy as np
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

WINDOW_SIZE = 1200  # 20 minutos en segundos
STEP_SIZE = 1200    # Paso de 20 minutos
DATA_TRAIN_DIR = f"data_per_{DATA_STRATEGY}_strategy/data/train"
DATA_TEST_DIR = f"data_per_{DATA_STRATEGY}_strategy/data/test"
DATA_VALID_DIR = f"data_per_{DATA_STRATEGY}_strategy/data/valid"

MODEL_SAVE_PATH = "model/trained_model.pkl"
REPORT_DIR = "reports"

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

def clean_labels(labels):
    labels = pd.Series(labels).astype(str).fillna("Desconocido").replace("nan", "Desconocido").values
    #print("Valores únicos después de transformar:", set(labels))
    return labels

def filter_unknown_labels(labels, known_labels):
    """Filtra etiquetas desconocidas que no están en known_labels"""
    return np.array([label if label in known_labels else 'Desconocido' for label in labels])

def create_windows(features, labels, window_size, step):
    X, y = [], []
    for i in range(0, len(features) - window_size + 1, step):
        X_window = features[i:i+window_size]
        y_window = labels[i:i+window_size]
        label = max(set(y_window), key=list(y_window).count)
        X.append(X_window.T)
        y.append(label)
    
    print(f"Total de ventanas creadas: {len(X)}")
    return np.array(X), np.array(y)

def evaluate_model(model, X, y, label_encoder, split_type="Train"):
    """Genera métricas detalladas y visualizaciones"""
    y_pred = model.predict(X)
    print(f"\n{split_type} Classification Report:")
    print(classification_report(y, y_pred, target_names=label_encoder.classes_, labels=range(len(label_encoder.classes_))))
    
    cm = confusion_matrix(y, y_pred, labels=range(len(label_encoder.classes_)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plt.title(f"{split_type} Confusion Matrix")
    plt.savefig(f"{REPORT_DIR}/{split_type.lower()}_confusion_matrix.png")
    plt.close()
    
    return accuracy_score(y, y_pred)

def predict_blocks(model, rocket, scaler, label_encoder, data, window_size, step_size):
    """Predice bloques de datos de 20 minutos"""
    X_scaled = scaler.transform(data)
    X_windows, _ = create_windows(X_scaled, np.zeros(len(X_scaled)), window_size, step_size)
    X_transformed = rocket.transform(X_windows)
    y_pred = model.predict(X_transformed)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    return y_pred_labels

# Crear directorios necesarios
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# Cargar y preparar datos de entrenamiento
print("Cargando datos de entrenamiento...")
X_train_raw, y_train_raw, df_train = load_and_preprocess_data(DATA_TRAIN_DIR)
y_train_raw = clean_labels(y_train_raw)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train_raw)

# Cargar y procesar datos de validación
print("Cargando datos de validación...")
X_valid_raw, y_valid_raw, _ = load_and_preprocess_data(DATA_VALID_DIR)
y_valid_raw = clean_labels(y_valid_raw)
X_valid_scaled = scaler.transform(X_valid_raw)
y_valid_encoded = le.transform(y_valid_raw)
X_valid_windows, y_valid_windows = create_windows(X_valid_scaled, y_valid_encoded, WINDOW_SIZE, STEP_SIZE)

# Crear ventanas para entrenamiento
print("Creando ventanas temporales de entrenamiento...")
X_train_windows, y_train_windows = create_windows(X_train_scaled, y_train_encoded, WINDOW_SIZE, STEP_SIZE)

# Verificar la distribución de clases en las ventanas
class_counts = Counter(y_train_windows)

# Ajustar k_neighbors para SMOTE
k_neighbors = min(5, min(class_counts.values()) - 1)

# Aplicar SMOTE
print("Aplicando SMOTE para balancear clases...")
smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_windows.reshape(X_train_windows.shape[0], -1), y_train_windows)
X_train_resampled = X_train_resampled.reshape(X_train_resampled.shape[0], 3, WINDOW_SIZE)

# Entrenar modelo Rocket
print("Entrenando modelo Rocket...")
rocket = Rocket(num_kernels=8000, random_state=42)
rocket.fit(X_train_resampled)
X_train_transformed = rocket.transform(X_train_resampled)
X_valid_transformed = rocket.transform(X_valid_windows)

# Entrenar clasificador con LightGBM
print("Entrenando clasificador con LightGBM...")
classifier = LGBMClassifier(
    n_estimators=300,  # Aumenta la cantidad de árboles
    learning_rate=0.05,  # Reduce la tasa de aprendizaje
    num_leaves=50,  # Controla la complejidad del modelo
    max_depth=7,  # Evita árboles demasiado profundos
    subsample=0.8,  # Reduce overfitting
    colsample_bytree=0.9,  # Usa 90% de las features en cada árbol
    reg_alpha=0.1,  # Regularización L1
    reg_lambda=0.1,  # Regularización L2
    random_state=42
)
classifier.fit(X_train_transformed, y_train_resampled)

# Evaluación
train_acc = evaluate_model(classifier, X_train_transformed, y_train_resampled, le, "Train")
valid_acc = evaluate_model(classifier, X_valid_transformed, y_valid_windows, le, "Validation")

test_acc = None
if os.path.exists(DATA_TEST_DIR) and os.listdir(DATA_TEST_DIR):
    try:
        print("\nEvaluando con datos de test...")
        X_test_raw, y_test_raw, _ = load_and_preprocess_data(DATA_TEST_DIR)
        y_test_raw = clean_labels(y_test_raw)
        y_test_raw = filter_unknown_labels(y_test_raw, le.classes_)
        X_test_scaled = scaler.transform(X_test_raw)
        y_test_encoded = le.transform(y_test_raw)
        X_test_windows, y_test_windows = create_windows(X_test_scaled, y_test_encoded, WINDOW_SIZE, STEP_SIZE)
        X_test_transformed = rocket.transform(X_test_windows)
        test_acc = evaluate_model(classifier, X_test_transformed, y_test_windows, le, "Test")
    except Exception as e:
        print(f"Error al evaluar con datos de test: {e}")

# Reporte final de métricas
print("\nResumen de Métricas:")
print(f"Train Accuracy: {train_acc:.3f}")
print(f"Validation Accuracy: {valid_acc:.3f}")
if test_acc is not None:
    print(f"Test Accuracy: {test_acc:.3f}")

# Guardar modelo y metadatos
joblib.dump({
    'rocket': rocket,
    'classifier': classifier,
    'scaler': scaler,
    'label_encoder': le,
    'window_size': WINDOW_SIZE,
    'metrics': {
        'train_accuracy': train_acc,
        'validation_accuracy': valid_acc,
        'test_accuracy': test_acc
    }
}, MODEL_SAVE_PATH)

print(f"\nModelo y reportes guardados en {MODEL_SAVE_PATH} y {REPORT_DIR}")