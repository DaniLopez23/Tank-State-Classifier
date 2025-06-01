import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sktime.transformations.panel.rocket import Rocket
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, f1_score, precision_score, recall_score)
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import StratifiedKFold
from scipy import stats

# Configuración
DATA_STRATEGY = "5_second"

# Configuración de ventanas con más overlapping
WINDOW_CONFIG = {
    "second": {"window_size": 1800, "step_size": 600}, 
    "minute": {"window_size": 30, "step_size": 10},       
    "5_second": {"window_size": 360, "step_size": 360}    
}

WINDOW_SIZE = WINDOW_CONFIG[DATA_STRATEGY]["window_size"]
STEP_SIZE = WINDOW_CONFIG[DATA_STRATEGY]["step_size"]

# Directorios
DATA_TRAIN_DIR = f"data_per_{DATA_STRATEGY}_strategy/data/train"
DATA_TEST_DIR = f"data_per_{DATA_STRATEGY}_strategy/data/test"
DATA_VALID_DIR = f"data_per_{DATA_STRATEGY}_strategy/data/valid"

MODEL_SAVE_PATH = f"data_per_{DATA_STRATEGY}_strategy/model/trained_model-simple.pkl"
REPORT_DIR = f"data_per_{DATA_STRATEGY}_strategy/reports"

# Hiperparámetros optimizados
ROCKET_KERNELS = {
    "second": 20000,
    "minute": 15000,
    "5_second": 6000
}

# Ridge con mejor configuración
RIDGE_CONFIG = {
    "alphas": np.logspace(5, 12, 100),  # Rango más enfocado
    "class_weight": "balanced",         # Importante para clases desbalanceadas
    "cv": StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    "scoring": 'f1_weighted'
}

NUM_KERNELS = ROCKET_KERNELS[DATA_STRATEGY]

def load_and_preprocess_data(folder_path):
    """Carga y preprocesa datos con limpieza básica pero efectiva"""
    all_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
    dfs = []
    
    for file in all_files:
        print(f"Processing {file}...")
        df = pd.read_csv(os.path.join(folder_path, file), parse_dates=["DateTime"])
        
        # Manejo de valores faltantes
        if df.isnull().any().any():
            print(f"Valores faltantes en {file}: {df.isnull().sum().sum()}")
            df.interpolate(method='linear', inplace=True)
            df.ffill(inplace=True)
            df.bfill(inplace=True)
        
        # Limpieza simple de outliers extremos
        numeric_cols = ["AccelX", "Surface temperature (ºC)", "Over surface temperature (ºC)"]
        for col in numeric_cols:
            Q1 = df[col].quantile(0.01)
            Q99 = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=Q1, upper=Q99)
        
        dfs.append(df)
    
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Solo las 3 características originales
    features = full_df[["AccelX", "Surface temperature (ºC)", "Over surface temperature (ºC)"]]
    labels = full_df["ETIQUETA"]

    print(f"\nTotal de muestras: {len(full_df)}")
    print("Distribución de clases:")
    class_dist = labels.value_counts(normalize=True)
    for label, pct in class_dist.items():
        print(f"  {label}: {pct:.2%} ({labels.value_counts()[label]} muestras)")
    
    return features.values, labels.values, full_df

def create_windows_smart(features, labels, window_size, step):
    """Ventanas con mejor estrategia de etiquetado para reducir confusión"""
    X, y = [], []
    n_samples = len(features)
    
    for i in range(0, n_samples - window_size + 1, step):
        end_idx = i + window_size
        X_window = features[i:end_idx]
        y_window = labels[i:end_idx]
        
        # Estrategia de etiquetado más inteligente
        unique_labels, counts = np.unique(y_window, return_counts=True)
        
        # Si hay una clase muy dominante (>70%), usar esa
        max_count_idx = np.argmax(counts)
        dominant_ratio = counts[max_count_idx] / len(y_window)
        
        if dominant_ratio >= 0.7:
            label = unique_labels[max_count_idx]
        else:
            # Para casos ambiguos, priorizar el centro de la ventana
            # que suele ser más representativo del estado real
            center_quarter = window_size // 4
            center_portion = y_window[center_quarter:-center_quarter]
            if len(center_portion) > 0:
                label = max(set(center_portion), key=list(center_portion).count)
            else:
                label = unique_labels[max_count_idx]
        
        # Normalización estándar
        scaler = StandardScaler()
        X_window_scaled = scaler.fit_transform(X_window)
        
        X.append(X_window_scaled.T)
        y.append(label)
    
    print(f"\nVentanas creadas: {len(X)}")
    print(f"Tamaño de ventana: {window_size} | Paso: {step}")
    
    # Distribución de clases en ventanas
    y_array = np.array(y)
    print("Distribución de clases en ventanas:")
    unique, counts = np.unique(y_array, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  {label}: {count} ventanas ({count/len(y):.2%})")
    
    return np.array(X), np.array(y)

def evaluate_model(model, X, y, label_encoder, split_type="Train"):
    """Evaluación enfocada en identificar confusiones específicas"""
    print(f"\nEvaluando {split_type} con datos de forma: {X.shape}")
    
    y_pred = model.predict(X)
    
    print(f"\n{split_type} Classification Report:")
    print(classification_report(y, y_pred, target_names=label_encoder.classes_, digits=4))
    
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'f1_weighted': f1_score(y, y_pred, average='weighted'),
        'f1_macro': f1_score(y, y_pred, average='macro'),
        'precision_macro': precision_score(y, y_pred, average='macro'),
        'recall_macro': recall_score(y, y_pred, average='macro')
    }
    
    # Matriz de confusión
    cm = confusion_matrix(y, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt=".3f", cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f"{split_type} Normalized Confusion Matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/{split_type.lower()}_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Análisis de errores específicos
    print(f"\n{split_type} - Principales confusiones:")
    error_pairs = []
    for i in range(len(label_encoder.classes_)):
        for j in range(len(label_encoder.classes_)):
            if i != j and cm[i, j] > 0:
                error_rate = cm[i, j] / cm[i].sum() * 100
                error_pairs.append((label_encoder.classes_[i], label_encoder.classes_[j], 
                                  cm[i, j], error_rate))
    
    error_pairs.sort(key=lambda x: x[3], reverse=True)  # Ordenar por porcentaje de error
    for true_label, pred_label, count, error_rate in error_pairs[:5]:
        print(f"  {true_label} → {pred_label}: {count} casos ({error_rate:.1f}%)")
    
    return metrics

# Crear directorios
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# Cargar datos de entrenamiento
print("\n" + "="*50)
print("CARGANDO DATOS DE ENTRENAMIENTO")
print("="*50)
X_train_raw, y_train_raw, df_train = load_and_preprocess_data(DATA_TRAIN_DIR)

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train_raw)

# Crear ventanas de entrenamiento
print("Creando ventanas de entrenamiento...")
X_train_windows, y_train_windows = create_windows_smart(X_train_raw, y_train_encoded, WINDOW_SIZE, STEP_SIZE)

# Crear ventanas de validación
print("\n" + "="*50)
print("CARGANDO DATOS DE VALIDACIÓN")
print("="*50)
X_valid_raw, y_valid_raw, _ = load_and_preprocess_data(DATA_VALID_DIR)
y_valid_encoded = le.transform(y_valid_raw)
X_valid_windows, y_valid_windows = create_windows_smart(X_valid_raw, y_valid_encoded, WINDOW_SIZE, STEP_SIZE)

# Entrenar ROCKET
print("\n" + "="*50)
print("ENTRENANDO ROCKET")
print("="*50)
print(f"Configurando ROCKET con {NUM_KERNELS} kernels...")
rocket = Rocket(num_kernels=NUM_KERNELS, random_state=42)
rocket.fit(X_train_windows)

# Transformar datos
print("Transformando datos...")
X_train_transformed = rocket.transform(X_train_windows)
X_valid_transformed = rocket.transform(X_valid_windows)

print(f"Datos transformados - Train: {X_train_transformed.shape}")
print(f"Datos transformados - Valid: {X_valid_transformed.shape}")

# Balancear clases tras ROCKET con SMOTE
print("\nBalanceando clases con SMOTE...")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_transformed, y_train_windows)
print("Distribución tras SMOTE:")
for idx, count in zip(*np.unique(y_train_bal, return_counts=True)):
    print(f"  {le.inverse_transform([idx])[0]}: {count}")

# Entrenar RidgeClassifierCV
print("\n" + "="*50)
print("ENTRENANDO RIDGECLASSIFIERCV")
print("="*50)
classifier = RidgeClassifierCV(**RIDGE_CONFIG)
classifier.fit(X_train_bal, y_train_bal)

print(f"Mejor alpha seleccionado: {classifier.alpha_:.2e}")
print(f"Score de validación cruzada: {classifier.best_score_:.4f}")

# Evaluación
print("\n" + "="*50)
print("EVALUACIÓN")
print("="*50)
train_metrics = evaluate_model(classifier, X_train_transformed, y_train_windows, le, "Train")
valid_metrics = evaluate_model(classifier, X_valid_transformed, y_valid_windows, le, "Validation")

# Evaluación en test si existe
test_metrics = None
if os.path.exists(DATA_TEST_DIR) and os.listdir(DATA_TEST_DIR):
    try:
        print("\n" + "="*50)
        print("EVALUACIÓN EN TEST")
        print("="*50)
        X_test_raw, y_test_raw, _ = load_and_preprocess_data(DATA_TEST_DIR)
        y_test_encoded = le.transform(y_test_raw)
        X_test_windows, y_test_windows = create_windows_smart(X_test_raw, y_test_encoded, WINDOW_SIZE, STEP_SIZE)
        X_test_transformed = rocket.transform(X_test_windows)
        test_metrics = evaluate_model(classifier, X_test_transformed, y_test_windows, le, "Test")
    except Exception as e:
        print(f"Error en test: {str(e)}")

# Guardar modelo
joblib.dump({
    'rocket': rocket,
    'classifier': classifier,
    'label_encoder': le,
    'window_config': {
        'window_size': WINDOW_SIZE,
        'step_size': STEP_SIZE
    },
    'metrics': {
        'train': train_metrics,
        'validation': valid_metrics,
        'test': test_metrics
    }
}, MODEL_SAVE_PATH)

print("\n" + "="*50)
print("ENTRENAMIENTO COMPLETADO!")
print("="*50)
print(f"Modelo guardado en: {MODEL_SAVE_PATH}")