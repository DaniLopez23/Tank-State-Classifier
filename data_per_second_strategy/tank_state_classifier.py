"""
Clasificador de estados de tanque con múltiples archivos de entrada
Autor: Asistente IA
"""

import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import classification_report
from sktime.transformations.panel.rocket import Rocket
from joblib import dump, load
from datetime import datetime

def crear_ventanas(df, window_size, step, features):
    """Crea ventanas temporales verificando etiquetas consistentes"""
    X, y, tiempos = [], [], []
    labels = df['ETIQUETA'].values
    datetimes = df['DateTime'].values
    
    for i in range(0, len(df) - window_size + 1, step):
        end_idx = i + window_size
        
        # Verificar etiqueta consistente
        if len(np.unique(labels[i:end_idx])) > 1:
            continue
        
        # Extraer ventana
        ventana = df[features].iloc[i:end_idx].values.T  # (features, pasos)
        X.append(ventana)
        y.append(labels[i])
        tiempos.append(datetimes[end_idx - 1])  # Tiempo final de la ventana
    
    return np.array(X), np.array(y), np.array(tiempos)

def procesar_archivos(archivos, window_size, step, features):
    """Procesa múltiples archivos y crea ventanas"""
    X_total, y_total, t_total = [], [], []
    
    for archivo in archivos:
        try:
            df = pd.read_csv(archivo, parse_dates=['DateTime'])
            df = df.sort_values('DateTime').reset_index(drop=True)
            X, y, t = crear_ventanas(df, window_size, step, features)
            
            if len(X) > 0:
                X_total.append(X)
                y_total.append(y)
                t_total.append(t)
        except Exception as e:
            print(f"Error procesando {archivo}: {str(e)}")
    
    return (np.concatenate(X_total) if X_total else np.array([]),
            np.concatenate(y_total) if y_total else np.array([]),
            np.concatenate(t_total) if t_total else np.array([]))

def main():
    parser = argparse.ArgumentParser(description='Clasificador de estados de tanque')
    parser.add_argument('input_files', nargs='+', help='Archivos CSV de entrada')
    parser.add_argument('--window_size', type=int, default=1200, help='Tamaño de ventana en segundos')
    parser.add_argument('--step', type=int, default=1200, help='Paso entre ventanas')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proporción para test')
    parser.add_argument('--split_method', choices=['file', 'time', 'random'], default='time',
                       help='Método de división train/test')
    parser.add_argument('--cutoff_date', type=lambda d: datetime.strptime(d, '%Y-%m-%d'), 
                       help='Fecha de corte para división temporal (YYYY-MM-DD)')
    args = parser.parse_args()

    # Configuración
    config = {
        'features': ['AccelX', 'Over surface temperature (ºC)', 'Surface temperature (ºC)'],
        'random_state': 42,
        'model_file': 'modelo_tanque.joblib'
    }

    # 1. Cargar datos y crear ventanas según método de división
    print(f"Procesando {len(args.input_files)} archivos...")
    
    if args.split_method == 'file':
        # División por archivos
        train_files, test_files = train_test_split(
            args.input_files, 
            test_size=args.test_size, 
            random_state=config['random_state']
        )
        
        X_train, y_train, _ = procesar_archivos(train_files, args.window_size, args.step, config['features'])
        X_test, y_test, _ = procesar_archivos(test_files, args.window_size, args.step, config['features'])
        
    else:
        # Cargar todos los datos
        X_full, y_full, t_full = procesar_archivos(args.input_files, args.window_size, args.step, config['features'])
        
        if args.split_method == 'time':
            # Ordenar por tiempo
            sort_idx = np.argsort(t_full)
            X_full = X_full[sort_idx]
            y_full = y_full[sort_idx]
            
            # División temporal
            split_idx = int(len(X_full) * (1 - args.test_size))
            X_train, X_test = X_full[:split_idx], X_full[split_idx:]
            y_train, y_test = y_full[:split_idx], y_full[split_idx:]
            
        elif args.split_method == 'random':
            # División aleatoria
            X_train, X_test, y_train, y_test = train_test_split(
                X_full, y_full, 
                test_size=args.test_size, 
                random_state=config['random_state'],
                stratify=y_full
            )

    # 2. Preprocesamiento
    print("Preprocesando...")
    # Codificar etiquetas (usando todas las etiquetas)
    le = LabelEncoder()
    le.fit(np.concatenate([y_train, y_test]))
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    
    # Escalar datos (solo con train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[1])).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[1])).reshape(X_test.shape)
    
    # 3. Transformación ROCKET
    print("Aplicando ROCKET...")
    rocket = Rocket(num_kernels=1000, random_state=config['random_state'])
    rocket.fit(X_train_scaled)
    
    X_train_trans = rocket.transform(X_train_scaled)
    X_test_trans = rocket.transform(X_test_scaled)
    
    # 4. Entrenar y evaluar modelo
    print("Entrenando modelo...")
    model = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    model.fit(X_train_trans, y_train_enc)
    
    print("\nEvaluación del modelo:")
    print(f"Exactitud test: {model.score(X_test_trans, y_test_enc):.2f}")
    print(classification_report(y_test_enc, model.predict(X_test_trans), target_names=le.classes_))
    
    # 5. Guardar modelo
    dump({'model': model, 'rocket': rocket, 'scaler': scaler, 'encoder': le}, config['model_file'])
    print(f"\nModelo guardado en {config['model_file']}")

if __name__ == "__main__":
    main()