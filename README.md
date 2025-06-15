# Tank-State-Classifier

This repository contains the **complete pipeline for preprocessing, labeling, training, and validation of classification models for milk tank states**. The goal is to automatically identify the operational state of the tank (e.g., MAINTENANCE, MILKING, COOLING, CLEANING, EMPTY TANK) from sensor data.

---

## What does this repository include?

- **Data preprocessing:** Cleaning, interpolation of missing values, and outlier detection/correction.
- **Labeling:** Integration and alignment of sensor data with state labels.
- **Training and validation:** Creation of temporal windows, class balancing, model training, and evaluation.
- **State classification model:** Based on state-of-the-art technologies for time series and multiclass classification.

---

## Processing strategies

The repository supports **three temporal granularity strategies** to adapt the model to different needs and sensor sampling rates:

1. **second:**  
   - 30-minute windows with data every second.
   - Maximum temporal resolution.
2. **5_second:**  
   - 30-minute windows with data every 5 seconds.
   - Balance between resolution and data size.
3. **minute:**  
   - 30-minute windows with data every minute.
   - Useful for long-term analysis or very large datasets.

Each strategy has its own window size and step configuration, adapting the model to the sensor sampling frequency.

---

## Initial datasets

- **acceldata:**  
  Acceleration data (e.g., X axis) collected from the tank.
- **tempdata:**  
  Surface and over-surface temperature data from the tank.

Both datasets are integrated and synchronized by timestamp to form the final dataset used for training and validation.

---

## State classification model

The classification pipeline uses the following technologies:

- **ROCKET (RandOm Convolutional KErnel Transform):**  
  Time series transformation technique that applies thousands of random convolutions to extract robust and efficient features.
- **RidgeClassifierCV:**  
  Linear classifier with L2 regularization and cross-validation to select the best regularization parameter (`alpha`). Handles imbalanced classes and is efficient for multiclass problems.
- **SMOTE:**  
  Oversampling technique to balance minority classes by generating synthetic examples.
- **scikit-learn, sktime, imbalanced-learn:**  
  Main libraries for machine learning, time series manipulation, and class balancing.

---

## Repository structure

- `rocket_train_model.py`: Main training and validation script.
- `rocket_api.py`: API to serve the trained model and make predictions.
- `data_per_*_strategy/`: Folders with processed data for each strategy.
- `reports/`: Evaluation reports and metrics.
- `test_api.py`: Script to test the API with new data.

---

## Author

Daniel López Paredes