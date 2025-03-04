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

# Configuración
WINDOW_SIZE = 1200  # 20 minutos en segundos
DATA_TRAIN_DIR = "data/train"
DATA_TEST_DIR = "data/test"
MODEL_SAVE_PATH = "model/trained_model.pkl"
REPORT_DIR = "reports"

def load_and_preprocess_data(folder_path):
    """Carga y preprocesa todos los CSVs en un directorio"""
    all_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
    dfs = []
    
    for file in all_files:
        df = pd.read_csv(os.path.join(folder_path, file), parse_dates=["DateTime"])
        dfs.append(df)
    
    full_df = pd.concat(dfs)
    features = full_df[["AccelX", "Surface temperature (ºC)", "Over surface temperature (ºC)"]]
    labels = full_df["ETIQUETA"]
    
    return features.values, labels.values, full_df

def create_windows(features, labels, window_size, step=200):
    X, y = [], []
    for i in range(0, len(features) - window_size + 1, step):
        X_window = features[i:i+window_size]
        y_window = labels[i:i+window_size]
        label = max(set(y_window), key=list(y_window).count)
        X.append(X_window.T)
        y.append(label)
    return np.array(X), np.array(y)

def evaluate_model(model, X, y, label_encoder, split_type="Train"):
    """Genera métricas detalladas y visualizaciones"""
    y_pred = model.predict(X)
    print(f"\n{split_type} Classification Report:")
    print(classification_report(y, y_pred, target_names=label_encoder.classes_))
    
    # Matriz de confusión
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plt.title(f"{split_type} Confusion Matrix")
    plt.savefig(f"{REPORT_DIR}/{split_type.lower()}_confusion_matrix.png")
    plt.close()
    
    return accuracy_score(y, y_pred)


# Crear directorios necesarios
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# Cargar y preparar datos
print("Cargando datos de entrenamiento...")
X_train_raw, y_train_raw, df_train = load_and_preprocess_data(DATA_TRAIN_DIR)

# Mostrar muestra de datos antes del preprocesamiento
print("\nMuestra de datos de entrenamiento:")
print(df_train.head())

# Preprocesamiento
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train_raw)

# Crear ventanas
print("Creando ventanas temporales...")
X_windows, y_windows = create_windows(X_train_scaled, y_train_encoded, WINDOW_SIZE)

# Split (90% entrenamiento, 10% validación)
split_idx = int(0.9 * len(X_windows))
X_train, X_val = X_windows[:split_idx], X_windows[split_idx:]
y_train, y_val = y_windows[:split_idx], y_windows[split_idx:]

# Entrenar modelo Rocket
print("Entrenando modelo Rocket...")
rocket = Rocket(num_kernels=5000, random_state=42)
rocket.fit(X_train)
X_train_transformed = rocket.transform(X_train)
X_val_transformed = rocket.transform(X_val)

# Entrenar clasificador con LightGBM
print("Entrenando clasificador con LightGBM...")
classifier = LGBMClassifier(n_estimators=50, learning_rate=0.05, random_state=42)
classifier.fit(X_train_transformed, y_train)

# Evaluación
train_acc = evaluate_model(classifier, X_train_transformed, y_train, le, "Train")
val_acc = evaluate_model(classifier, X_val_transformed, y_val, le, "Validation")

# Evaluación con datos de test si existen
test_acc = None
if os.path.exists(DATA_TEST_DIR) and os.listdir(DATA_TEST_DIR):
    print("\nEvaluando con datos de test...")
    X_test_raw, y_test_raw, _ = load_and_preprocess_data(DATA_TEST_DIR)
    
    # Preprocesar con scaler y encoder existentes
    X_test_scaled = scaler.transform(X_test_raw)
    try:
        y_test_encoded = le.transform(y_test_raw)
    except ValueError as e:
        print(f"Advertencia: {e}. Filtrando clases desconocidas...")
        mask = np.isin(y_test_raw, le.classes_)
        X_test_scaled = X_test_scaled[mask]
        y_test_raw = y_test_raw[mask]
        y_test_encoded = le.transform(y_test_raw)
    
    X_test_windows, y_test_windows = create_windows(X_test_scaled, y_test_encoded, WINDOW_SIZE)
    X_test_transformed = rocket.transform(X_test_windows)
    test_acc = evaluate_model(classifier, X_test_transformed, y_test_windows, le, "Test")

# Reporte final de métricas
print("\nResumen de Métricas:")
print(f"Train Accuracy: {train_acc:.3f}")
print(f"Validation Accuracy: {val_acc:.3f}")
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
        'validation_accuracy': val_acc,
        'test_accuracy': test_acc
    }
}, MODEL_SAVE_PATH)

print(f"\nModelo y reportes guardados en {MODEL_SAVE_PATH} y {REPORT_DIR}")
