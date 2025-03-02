import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sktime.transformations.panel.rocket import Rocket
import lightgbm as lgbm
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils.class_weight import compute_class_weight

CONFIG = {
    "train_files": [
        "labeled_data/labeled_data_2024-08-23.csv",
        "labeled_data/labeled_data_2024-08-24.csv",
        "labeled_data/labeled_data_2024-08-25.csv"
        ],  # Puedes agregar más archivos
    "test_files": [
        "labeled_data/labeled_data_2024-08-26.csv",
        "labeled_data/labeled_data_2024-08-27.csv"
        ],
    "window_size": 1200,  # 20 minutos (1200 segundos si es 1 Hz)
    "num_kernels": 500,
    "random_state": 42,
    "invalid_value": -127.0,
    "output": {
        "model_path": "tank_state_model.pkl",
        "encoder_path": "label_encoder.pkl",
        "plot_path": "confusion_matrix.png"
    }
}

def load_and_preprocess(files):
    """ Cargar y procesar múltiples archivos en un solo DataFrame """
    df_list = []
    for file in files:
        df = pd.read_csv(file)
        df["DateTime"] = pd.to_datetime(df["DateTime"])
        df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)
    
    # Ordenar por tiempo y manejar valores inválidos
    df = df.sort_values("DateTime")
    sensores = ["AccelX", "Over surface temperature (ºC)", "Surface temperature (ºC)"]
    for sensor in sensores:
        df[sensor] = df[sensor].replace(CONFIG["invalid_value"], np.nan)
    df = df.ffill().bfill()
    
    return df

df_train = load_and_preprocess(CONFIG["train_files"])
df_test = load_and_preprocess(CONFIG["test_files"])

# Codificar etiquetas
label_encoder = LabelEncoder()
df_train["ETIQUETA"] = label_encoder.fit_transform(df_train["ETIQUETA"].astype(str))
df_test["ETIQUETA"] = label_encoder.transform(df_test["ETIQUETA"].astype(str))

def create_windows(df, sensores, window_size):
    X, y = [], []
    timestamps = df["DateTime"].values
    
    for i in range(0, len(df) - window_size, window_size):
        ventana = df.iloc[i:i + window_size]
        
        if len(ventana) < window_size:
            continue
        
        ventana_valores = ventana[sensores].values.T
        ventana_norm = (ventana_valores - np.mean(ventana_valores, axis=1, keepdims=True)) / (np.std(ventana_valores, axis=1, keepdims=True) + 1e-8)
        
        X.append(ventana_norm.astype(np.float32))
        y.append(ventana["ETIQUETA"].mode()[0])  # Tomar la etiqueta más frecuente en la ventana
    
    return np.array(X), np.array(y)

sensores = ["AccelX", "Over surface temperature (ºC)", "Surface temperature (ºC)"]

X_train, y_train = create_windows(df_train, sensores, CONFIG["window_size"])
X_test, y_test = create_windows(df_test, sensores, CONFIG["window_size"])

# Balanceo de clases
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
weights_dict = {i: weight for i, weight in enumerate(class_weights)}

# Modelo
model = make_pipeline(
    Rocket(num_kernels=CONFIG["num_kernels"], random_state=CONFIG["random_state"]),
    lgbm.LGBMClassifier(
        num_leaves=31,
        max_depth=5,
        learning_rate=0.05,
        n_estimators=100,
        class_weight=weights_dict,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=CONFIG["random_state"]
    )
)

print ("X_train shape: ", X_train.shape)
print ("y_train shape: ", y_train.shape)
print ("X_test shape: ", X_test.shape)
print ("y_test shape: ", y_test.shape)

# Entrenamiento
model.fit(X_train, y_train)

# Evaluación
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Reportes de clasificación
print("\nResultados en TRAIN:")
print(classification_report(label_encoder.inverse_transform(y_train), label_encoder.inverse_transform(y_pred_train), zero_division=0))

print("\nResultados en TEST:")
print(classification_report(label_encoder.inverse_transform(y_test), label_encoder.inverse_transform(y_pred_test), zero_division=0))

# Gráfica de la matriz de confusión
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

ConfusionMatrixDisplay.from_predictions(
    label_encoder.inverse_transform(y_train),
    label_encoder.inverse_transform(y_pred_train),
    display_labels=label_encoder.classes_,
    cmap="Blues",
    xticks_rotation=45,
    values_format=".0f",
    ax=ax[0]
)
ax[0].set_title("Matriz de Confusión - Train")

ConfusionMatrixDisplay.from_predictions(
    label_encoder.inverse_transform(y_test),
    label_encoder.inverse_transform(y_pred_test),
    display_labels=label_encoder.classes_,
    cmap="Blues",
    xticks_rotation=45,
    values_format=".0f",
    ax=ax[1]
)
ax[1].set_title("Matriz de Confusión - Test")

plt.tight_layout()
plt.savefig(CONFIG["output"]["plot_path"])
plt.show()

# Guardar modelos
joblib.dump(model, CONFIG["output"]["model_path"])
joblib.dump(label_encoder, CONFIG["output"]["encoder_path"])

print(f"\nModelo guardado en {CONFIG['output']['model_path']}")
