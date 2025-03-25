import os
import pandas as pd

def process_csv_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for file in os.listdir(input_dir):
        if file.endswith(".csv"):
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, file)
            
            df = pd.read_csv(input_path)
            
            if 'DateTime' not in df.columns:
                print(f"Skipping {file}: No 'DateTime' column found.")
                continue
            
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            df = df.sort_values(by='DateTime')
            
            df_resampled = df[df['DateTime'].dt.second % 5 == 0]
            
            df_resampled.to_csv(output_path, index=False)
            print(f"Processed {file} -> {output_path}")

# Configura los directorios de entrada y salida
input_directory = "data_per_second_strategy/labeled_data"
output_directory = "data_per_5_second_strategy/labeled_data/"

process_csv_files(input_directory, output_directory)
