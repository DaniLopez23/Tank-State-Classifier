import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sktime.classification.kernel_based import RocketClassifier
from sktime.split import TemporalTrainTestSplit

FILE_DATE = "2024-08-24"

CONFIG = {
    "data_path": f"labeled_data/labeled_data_{FILE_DATE}.csv",
    "window_size": 60,  # 1 minuto = 60 muestras (1 muestra/segundo)
    "test_size": 0.2,  # 20% para test
    "num_kernels": 10000,  # Aumentar para mejor precisión (requiere más RAM)
    "random_state": 42,
    "output": {
        "model_path": "tank_state_model.pkl",
        "encoder_path": "label_encoder.pkl",
        "plot_path": "confusion_matrix.png"
    }
}

# Cargar datos
df = pd.read_csv(CONFIG["data_path"])
df["DateTime"] = pd.to_datetime(df["DateTime"])
df = df.sort_values("DateTime")

# Manejar valores inválidos
print("\nValores únicos antes de limpieza:")
print(df.nunique())

df["Submerged temperature (ºC)"] = df["Submerged temperature (ºC)"].replace(-127.0, np.nan)
df = df.ffill().bfill()  # Rellenar NaN

# Codificar etiquetas
label_encoder = LabelEncoder()
df["ETIQUETA"] = label_encoder.fit_transform(df["ETIQUETA"])

# Crear ventanas temporales
n_muestras = len(df) // CONFIG["window_size"]
sensores = ["AccelX", "Over surface temperature (ºC)", 
           "Submerged temperature (ºC)", "Surface temperature (ºC)"]

X = np.zeros((n_muestras, len(sensores), CONFIG["window_size"]))
y = []

for i in range(n_muestras):
    start = i * CONFIG["window_size"]
    end = start + CONFIG["window_size"]
    
    ventana = df[sensores].iloc[start:end]
    X[i] = ventana.values.T
    
    etiqueta = df["ETIQUETA"].iloc[start:end].mode()[0]
    y.append(etiqueta)

X = X.astype("float32")
y = np.array(y)

# Filtrar ventanas incompletas
valid_windows = ~np.isnan(X).any(axis=(1,2))
X, y = X[valid_windows], y[valid_windows]

# %% 4. Entrenamiento del modelo
# Split temporal
splitter = TemporalTrainTestSplit(test_size=CONFIG["test_size"])
X_train, X_test, y_train, y_test = splitter.split(X, y)

# Inicializar y entrenar modelo
model = RocketClassifier(
    num_kernels=CONFIG["num_kernels"],
    random_state=CONFIG["random_state"],
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluación
y_pred = model.predict(X_test)
report = classification_report(
    label_encoder.inverse_transform(y_test),
    label_encoder.inverse_transform(y_pred),
    target_names=label_encoder.classes_
)
print("\nClassification Report:")
print(report)

# %% 5. Visualización y guardado
# Matriz de confusión
fig, ax = plt.subplots(figsize=(12, 10))
ConfusionMatrixDisplay.from_predictions(
    label_encoder.inverse_transform(y_test),
    label_encoder.inverse_transform(y_pred),
    display_labels=label_encoder.classes_,
    ax=ax,
    cmap="Blues",
    xticks_rotation=45,
    values_format=".0f"
)
plt.title("Estados del Tanque - Matriz de Confusión")
plt.tight_layout()
plt.savefig(CONFIG["output"]["plot_path"])
plt.show()

# Guardar recursos
joblib.dump(model, CONFIG["output"]["model_path"])
joblib.dump(label_encoder, CONFIG["output"]["encoder_path"])

print(f"\nProceso completado. Modelo guardado en {CONFIG['output']['model_path']}")