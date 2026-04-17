# -*- coding: utf-8 -*-
"""
AAI Framework for Subseasonal-to-Seasonal (S2S) Forecast Bias Correction.
PyTorch & Classical Version.

UNIFIED DL & CLASSICAL VERSION (PyTorch):
1. Uses Randomized Search Cross-Validation for adaptive hyperparameter tuning
   for all models (RF, XGB, SVM, LSTM, CNN, Transformer).
2. Deep Learning models are now integrated using PyTorch.
"""

################################ Libraries ####################################
import os
import numpy as np
import pandas as pd
import sqlite3
import datetime
import warnings

# --- PyTorch Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.base import BaseEstimator, RegressorMixin

# --- Classical ML & Sklearn Imports ---
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, make_scorer
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Set default device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################## Global Variables #################################
VAR_NAMES = ['TMax']
# --- AAI Settings ---
WINDOW_SIZE = 46
TEST_WINDOW = 6
RANDOM_SEARCH_ITER = 25 # Number of HP combinations for RandomizedSearchCV

# The expected columns are: Model, Year, Month, Step, Latitude, Longitude, S2S, Obs
CATEGORICAL_FEATURES = []
Data_Type = 1  # If Data_Type is 'Random Datasets' use '0', if you have 'Own Datasets' use '1'
OUTPUT_DIR = r"C:\Users\venky\Dropbox\IISC\Ideas\Warm_Season_S2S\Manuscript\Submission\JoHx\JoHx_Updated_S2S\Revision_1\Models"
############################## PyTorch Utilities ###############################

class BiasCorrectionDataset(Dataset):
    """Custom PyTorch Dataset for loading features (X) and target (y)."""
    def __init__(self, X, y=None):
        # Convert NumPy arrays to PyTorch Tensors, ensuring float32
        self.X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            # Reshape y to (N, 1) for MSE loss in PyTorch
            self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

class TorchRegressor(BaseEstimator, RegressorMixin):
    """Custom scikit-learn compatible wrapper for PyTorch models."""
    def __init__(self, model_builder, input_shape=None, criterion=nn.MSELoss,
                 optimizer=optim.Adam, lr=0.001, batch_size=32, epochs=10,
                 **model_params):
        # Hyperparameters for the wrapper and training loop
        self.model_builder = model_builder
        self.input_shape = input_shape
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_params = model_params
        
        # Internal state
        self.model = None
        self.is_fit = False

    def fit(self, X, y, **kwargs):
        # Set input_shape if not set (for first model creation)
        if self.input_shape is None:
             self.input_shape = X.shape[1:]

        # 1. Initialize Model
        # model_params contains lstm_units, dense_units, etc.
        self.model = self.model_builder(input_shape=self.input_shape, **self.model_params).to(DEVICE)
        
        # 2. Setup training components
        optimizer = self.optimizer(self.model.parameters(), lr=self.lr)
        criterion = self.criterion()

        # 3. Prepare Data
        dataset = BiasCorrectionDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 4. Training Loop
        self.model.train()
        for epoch in range(self.epochs):
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()

        self.is_fit = True
        return self

    def predict(self, X, **kwargs):
        if not self.is_fit:
            raise Exception("Model not fit yet!")

        self.model.eval()
        dataset = BiasCorrectionDataset(X)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        predictions = []
        with torch.no_grad():
            for X_batch in loader:
                X_batch = X_batch.to(DEVICE)
                output = self.model(X_batch)
                predictions.append(output.cpu().numpy())

        # Combine results, flatten and return numpy array
        predictions = np.concatenate(predictions, axis=0)
        return predictions.flatten()

    def get_params(self, deep=True):
        # Expose all parameters for RandomizedSearchCV
        params = super().get_params(deep=deep)
        params.update({
            'model_builder': self.model_builder,
            'input_shape': self.input_shape,
            'criterion': self.criterion,
            'optimizer': self.optimizer,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
        })
        # Add model-specific parameters
        params.update(self.model_params)
        return params

    def set_params(self, **params):
        # Update wrapper parameters and model-specific parameters (stored in model_params)
        if 'model_builder' in params: self.model_builder = params.pop('model_builder')
        if 'input_shape' in params: self.input_shape = params.pop('input_shape')
        if 'criterion' in params: self.criterion = params.pop('criterion')
        if 'optimizer' in params: self.optimizer = params.pop('optimizer')
        if 'lr' in params: self.lr = params.pop('lr')
        if 'batch_size' in params: self.batch_size = params.pop('batch_size') 
        if 'epochs' in params: self.epochs = params.pop('epochs')
        
        # All remaining params are model_params
        self.model_params.update(params)
        return self


############################## Utility Functions ###############################

def load_data(variable, Data_Type):
    """
    Load the data from the SQLite database, handle the categorical 'Model' feature,
    and calculate the target 'Bias'.
    """
    print(f"Loading and processing data for {variable} from S2S_India.db...")
    
    df = None
    if Data_Type == 1:
        try:
            conn = sqlite3.connect('S2S_India.db')
            # LIMIT applied here for debugging and memory safety
            # query = f"SELECT * FROM {variable} LIMIT 10000"
            query = "SELECT * FROM Tmax_Sorted"
            df = pd.read_sql_query(query, conn)
            conn.close()
            print(f"Successfully loaded {len(df)} rows (temporary limit).")
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            print("Using mock data as a fallback to ensure execution continuity.")
            
    if df is None or df.empty or Data_Type == 0:
        # FALLBACK: Minimal mock data generation if DB loading fails
        n_samples = 250
        dates = pd.date_range(start='2015-01-01', periods=n_samples, freq='W')
        df = pd.DataFrame({
            'Model': np.random.choice(['ECMWF', 'NCEP', 'UKMO'], size=n_samples),
            'Year': dates.year,
            'Month': dates.month,
            'Step': np.random.randint(1, 7, size=n_samples),
            'Latitude': np.random.uniform(8, 38, size=n_samples).round(2),
            'Longitude': np.random.uniform(68, 98, size=n_samples).round(2),
            'S2S': np.random.rand(n_samples) * 10,
            'Obs': np.random.rand(n_samples) * 10,
            'Unnamed: 0': np.arange(n_samples)
        })
        
    # Data Cleaning and Preprocessing
    if df.columns[0] in ['index', 'level_0', 'Unnamed: 0']:
        df = df.iloc[:, 1:].copy()
        
    numerical_cols = [col for col in df.columns if col not in CATEGORICAL_FEATURES]
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').round(2)
        
    df.dropna(inplace=True)
    
    # AAI Feature Engineering (Contextual Adaptivity: One-Hot Encoding)
    if len(CATEGORICAL_FEATURES) > 0:
        df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, prefix=CATEGORICAL_FEATURES)
    
    # AAI Target Variable: Calculate Bias
    df['Bias'] = df['Obs'] - df['S2S']
    
    print(f"Data prepared. Final feature count: {df.shape[1] - 2} (excluding Obs/Bias).")
    return df


def prepare_data(df):
    EXCLUDE_COLS = ['Obs', 'Bias']
    feature_cols = [col for col in df.columns if col not in EXCLUDE_COLS]
    
    X = df[feature_cols]
    y = df['Bias']
    S2S_raw = df['S2S']
    obs_raw = df['Obs']
    
    return X, y, S2S_raw, obs_raw

def reshape_for_dl(X, model_type):
    """Reshape 2D data (samples, features) to 3D for DL models."""
    if model_type in ['LSTM', 'Transformer']:
        # Treat feature vector as a sequence of length 1: (samples, timesteps=1, features)
        return X.reshape((X.shape[0], 1, X.shape[1]))
    elif model_type == 'CNN':
        # Treat features as the sequence dimension: (samples, steps=features, channels=1)
        return X.reshape((X.shape[0], X.shape[1], 1))
    return X # Should not happen

############################## Model Builder Functions (Classical) ###############################

def build_xgb_model():
    """Instantiate XGBoost template."""
    return XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=1)

def build_rf_model():
    """Instantiate Random Forest template."""
    return RandomForestRegressor(random_state=42, n_jobs=1)

def build_svm_model():
    """Instantiate SVM template. Scaling is handled externally."""
    return SVR(kernel='rbf')

############################## Model Builder Functions (PyTorch Deep Learning) ###############################

class LSTMModel(nn.Module):
    """PyTorch LSTM Model."""
    def __init__(self, input_shape, lstm_units=50, dense_units=20):
        super().__init__()
        # input_shape is (timesteps, features) -> (1, features)
        features = input_shape[-1]
        self.lstm = nn.LSTM(features, lstm_units, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.dense1 = nn.Linear(lstm_units, dense_units)
        self.output = nn.Linear(dense_units, 1)

    def forward(self, x):
        # x shape: (batch_size, 1, features)
        # Use only the final hidden state (h_n) for the dense layers
        _, (h_n, _) = self.lstm(x)
        # h_n shape: (1, batch_size, lstm_units) -> squeeze to (batch_size, lstm_units)
        x = h_n.squeeze(0)
        x = self.dropout(x)
        x = self.relu(self.dense1(x))
        return self.output(x)

def build_lstm_model(input_shape, lstm_units=50, dense_units=20, **kwargs):
    return LSTMModel(input_shape, lstm_units, dense_units)

class CNNModel(nn.Module):
    """PyTorch 1D CNN Model (equivalent to Conv1D/GlobalAveragePooling1D)."""
    def __init__(self, input_shape, filters=64, kernel_size=3, dense_units=20):
        super().__init__()
        # input_shape is (features, 1) in Keras style. channels=1
        features = input_shape[0]
        channels = input_shape[1] 

        # PyTorch Conv1d expects (N, C, L). We handle the transpose in forward().
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=filters, kernel_size=kernel_size)
        
        # GlobalAveragePooling1D equivalent
        self.avgpool = nn.AdaptiveAvgPool1d(1) 
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.dense1 = nn.Linear(filters, dense_units)
        self.output = nn.Linear(dense_units, 1)

    def forward(self, x):
        # Input x shape: (batch_size, features, 1)
        
        # Transpose for PyTorch Conv1D: (N, L, C) -> (N, C, L)
        x = x.transpose(1, 2) 

        x = self.relu(self.conv1(x))
        # x shape: (batch_size, filters, L')
        
        x = self.avgpool(x).squeeze(-1)
        # x shape: (batch_size, filters)

        x = self.dropout(x)
        x = self.relu(self.dense1(x))
        return self.output(x)

def build_cnn_model(input_shape, filters=64, kernel_size=3, dense_units=20, **kwargs):
    return CNNModel(input_shape, filters, kernel_size, dense_units)


class TransformerModel(nn.Module):
    """PyTorch simple Attention-based Model."""
    def __init__(self, input_shape, heads=2, ffn_dim=32):
        super().__init__()
        # input_shape is (timesteps, features) -> (1, features)
        d_model = input_shape[-1]
        
        # PyTorch's standard Transformer Encoder Layer
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=0.2,
            batch_first=True
        )
        # Use a single layer as in the original Keras code
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(d_model, 1) # Output dim is d_model (features) after flatten

    def forward(self, x):
        # x shape: (batch_size, 1, d_model)
        x = self.transformer_encoder(x)
        # x shape: (batch_size, 1, d_model)
        
        # Flatten from (N, 1, F) to (N, F)
        x = self.flatten(x) 
        x = self.dropout(x)
        return self.output(x)

def build_transformer_model(input_shape, heads=2, ffn_dim=32, **kwargs):
    return TransformerModel(input_shape, heads, ffn_dim)


############################## Evaluation Functions ##############################

def evaluate_model(model, X_test, y_test, S2S_raw_test, obs_test, model_name):
    """Evaluate Scikit-learn or TorchRegressor models."""
    
    is_dl = isinstance(model, TorchRegressor) or any(m in model_name for m in ['LSTM', 'CNN', 'Transformer'])

    if is_dl:
        # Reshape data for DL evaluation
        X_test_reshaped = reshape_for_dl(X_test, model_name)
        predicted_bias = model.predict(X_test_reshaped).flatten()
    else:
        # Classical models use 2D data
        predicted_bias = model.predict(X_test).flatten()
    
    # Apply the Adaptive Bias Correction
    testPredict_corrected = S2S_raw_test + predicted_bias
    
    rawScore = np.sqrt(mean_squared_error(obs_test, S2S_raw_test))
    testScore = np.sqrt(mean_squared_error(obs_test, testPredict_corrected))
    
    return rawScore, testScore

############################## Core Logic ######################################

def select_best_model(X, y, S2S_raw, obs_raw):
    """
    Selects the best model (classical or DL) by comparing performance across models tuned
    via Randomized Search Cross-Validation.
    """
    print("\n--- Selecting Best Initial Model via Randomized Search Tuning ---")
    
    # Use a fixed split for fair comparison (80% train / 20% test)
    X_train, X_test, y_train, y_test, S2S_train, S2S_test, obs_train, obs_test = train_test_split(
        X, y, S2S_raw, obs_raw, test_size=0.2, shuffle=False
    )
    
    all_results = []
    input_features = X.shape[1]
    
    # SVM and DL models require scaling for training and prediction
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # Prepare DL data (reshaped and scaled for DL models)
    # LSTM/Transformer uses (1, features)
    X_train_dl_scaled = reshape_for_dl(X_train_scaled, 'LSTM') 
    X_test_dl_scaled = reshape_for_dl(X_test_scaled, 'LSTM')  

    # CNN uses (features, 1)
    X_train_cnn_scaled = reshape_for_dl(X_train_scaled, 'CNN') 
    X_test_cnn_scaled = reshape_for_dl(X_test_scaled, 'CNN')
    
    # Define RMSE scorer for RandomizedSearchCV
    rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False, squared=False)
    
    # =============================================================================
    # 🔁 CLASSICAL MODELS (OPTIMIZED)
    # =============================================================================
    
    tuning_models = {
    
        'XGBoost': (
            build_xgb_model(), 
            X_train, 
            X_test, 
            {
                'n_estimators': [200, 400, 800],
                'learning_rate': [0.01, 0.03, 0.05],
                'max_depth': [3, 4, 5, 6],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'gamma': [0, 0.1, 0.3],
                'reg_alpha': [0, 0.1],
                'reg_lambda': [1, 5]
            }
        ),
    
        'RandomForest': (
            build_rf_model(), 
            X_train, 
            X_test,
            {
                'n_estimators': [200, 400, 800],
                'max_depth': [10, 20, None],
                'min_samples_leaf': [1, 2, 5],
                'min_samples_split': [2, 5],
                'max_features': ['sqrt']
            }
        ),
    
        'SVM': (
            build_svm_model(), 
            X_train_scaled, 
            X_test_scaled, 
            {
                'C': [0.5, 1, 10, 50],
                'gamma': ['scale', 0.01, 0.05],
                'epsilon': [0.01, 0.1]
            }
        )
    }
    
    # =============================================================================
    # 🧠 DEEP LEARNING MODELS (OPTIMIZED)
    # =============================================================================
    
    dl_models = {
    
        'LSTM': (
            TorchRegressor(
                model_builder=build_lstm_model, 
                input_shape=X_train_dl_scaled.shape[1:]
            ),
            X_train_dl_scaled, 
            X_test_dl_scaled,
            {
                'lstm_units': [32, 64, 128],
                'dense_units': [16, 32],
                'lr': [0.0005, 0.001],
                'epochs': [20],
                'batch_size': [16, 32]
            }
        ),
    
        'CNN': (
            TorchRegressor(
                model_builder=build_cnn_model, 
                input_shape=X_train_cnn_scaled.shape[1:]
            ),
            X_train_cnn_scaled, 
            X_test_cnn_scaled,
            {
                'filters': [32, 64],
                'kernel_size': [2, 3],
                'dense_units': [16, 32],
                'lr': [0.0005, 0.001],
                'epochs': [20],
                'batch_size': [16, 32]
            }
        ),
    
        'Transformer': (
            TorchRegressor(
                model_builder=build_transformer_model, 
                input_shape=X_train_dl_scaled.shape[1:]
            ),
            X_train_dl_scaled, 
            X_test_dl_scaled,
            {
                'heads': [2],
                'ffn_dim': [32, 64],
                'lr': [0.0005, 0.001],
                'epochs': [20],
                'batch_size': [16, 32]
            }
        )
    }
    
    # 🔥 MERGE
    tuning_models.update(dl_models)
    
    print(f"--- Hyperparameter Tuning ({RANDOM_SEARCH_ITER} iterations per model) ---")
    pbar = tqdm(tuning_models.items(), desc="Tuning Progress")

    for name, (model, X_tune_train, X_tune_test, param_dist) in pbar:
        # Update the progress bar description to show the current model
        # print(f"Tuning {name}...")
        with tqdm(total=RANDOM_SEARCH_ITER, desc=f"Tuning {name}") as pbar:
    
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_dist,
                n_iter=RANDOM_SEARCH_ITER,
                cv=3,
                scoring=rmse_scorer,
                random_state=42,
                # Force n_jobs=1 for PyTorch models as they cannot be reliably pickled across processes
                n_jobs=1 if isinstance(model, TorchRegressor) else 1,
                verbose=1
            )
            
            # Fit the search on the training data
            random_search.fit(X_tune_train, y_train)
            best_model = random_search.best_estimator_
            
            # Determine which data split to use for final evaluation
            X_eval = X_test_scaled if name in ['SVM', 'LSTM', 'CNN', 'Transformer'] else X_test
            
            raw_rmse, corrected_rmse = evaluate_model(
                best_model, X_eval, y_test, S2S_test, obs_test, name
            )
                
            print(f"  Best {name} Corrected RMSE: {corrected_rmse:.4f} (HPs: {random_search.best_params_})")
            all_results.append({
                'model_name': name, 'corrected_rmse': corrected_rmse, 
                'correction_model': best_model, 
                'scaler_X': (scaler_X if name in ['SVM', 'LSTM', 'CNN', 'Transformer'] else None)
            })
    
    # --- Final Selection ---
    best_result = min(all_results, key=lambda x: x['corrected_rmse'])
    
    print(f"\n✅ Best Model Selected: {best_result['model_name']} with Final Test RMSE {best_result['corrected_rmse']:.4f}")
    return best_result['model_name'], best_result['correction_model'], best_result['scaler_X']




def run_adaptive_correction(df, best_model_template, best_model_name, initial_scaler):
    """
    Adaptive correction using TIME-BASED sliding window (correct for S2S data).
    """

    # ==============================
    # 1. CREATE TIME INDEX
    # ==============================
    df = df.copy()
    df = df.sort_values(by=['Year','Month','Step','Latitude','Longitude']).reset_index(drop=True)

    df['time_id'] = df.groupby(['Year','Month','Step']).ngroup()

    time_steps = sorted(df['time_id'].unique())

    corrected_forecasts = np.full(len(df), np.nan)

    is_dl = best_model_name in ['LSTM', 'CNN', 'Transformer']
    is_scaled = best_model_name in ['SVM', 'LSTM', 'CNN', 'Transformer']

    print(f"\n--- Running TIME-BASED Adaptive Correction ({best_model_name}) ---")
    print(f"Total time steps: {len(time_steps)}")

    # ==============================
    # 2. SLIDING WINDOW OVER TIME
    # ==============================
    for i in range(0, len(time_steps) - WINDOW_SIZE, TEST_WINDOW):

        train_times = time_steps[i : i + WINDOW_SIZE]
        test_times  = time_steps[i + WINDOW_SIZE : i + WINDOW_SIZE + TEST_WINDOW]

        train_df = df[df['time_id'].isin(train_times)]
        test_df  = df[df['time_id'].isin(test_times)]

        if len(test_df) == 0:
            continue

        # ==============================
        # 3. PREPARE DATA
        # ==============================
        X_train, y_train, S2S_train, obs_train = prepare_data(train_df)
        X_test,  y_test,  S2S_test,  obs_test  = prepare_data(test_df)

        # ==============================
        # 4. SCALING (if required)
        # ==============================
        if is_scaled:
            scaler_X = StandardScaler()
            X_train = scaler_X.fit_transform(X_train)
            X_test  = scaler_X.transform(X_test)

        # ==============================
        # 5. RESHAPE FOR DL
        # ==============================
        if is_dl:
            X_train = reshape_for_dl(X_train, best_model_name)
            X_test  = reshape_for_dl(X_test, best_model_name)

        # ==============================
        # 6. RE-INITIALIZE MODEL
        # ==============================
        model = best_model_template.__class__(**best_model_template.get_params())

        # ==============================
        # 7. TRAIN
        # ==============================
        model.fit(X_train, y_train)

        # ==============================
        # 8. PREDICT BIAS
        # ==============================
        predicted_bias = model.predict(X_test).flatten()

        # ==============================
        # 9. APPLY CORRECTION
        # ==============================
        corrected_values = S2S_test + predicted_bias

        # ==============================
        # 10. STORE RESULTS
        # ==============================
        test_indices = test_df.index
        corrected_forecasts[test_indices] = corrected_values

        print(f"Window {i}: Train steps={len(train_times)}, Test steps={len(test_times)}")

    # ==============================
    # 11. FINAL EVALUATION
    # ==============================
    valid = ~np.isnan(corrected_forecasts)

    final_obs = df.loc[valid, 'Obs'].values
    final_s2s = df.loc[valid, 'S2S'].values
    final_corr = corrected_forecasts[valid]

    rmse_raw = np.sqrt(mean_squared_error(final_obs, final_s2s))
    rmse_corr = np.sqrt(mean_squared_error(final_obs, final_corr))

    skill = (rmse_raw - rmse_corr) / rmse_raw * 100

    print("\n-------------------------------------------------")
    print("      AAI ADAPTIVE BIAS CORRECTION RESULTS         ")
    print("-------------------------------------------------")
    print(f"Base Model Used: {best_model_name} (Adaptively Tuned)")
    print(f"Raw S2S Forecast RMSE: {rmse_raw:.4f}")
    print(f"AAI Corrected Forecast RMSE: {rmse_corr:.4f}")
    print(f"Skill Improvement: {skill:.2f}%")
    print("-------------------------------------------------")
    df['AAI_Corrected'] = corrected_forecasts

    return df

def save_results(df, variable):
    """Save the results (Mocked DB save)."""
    print(f"\nSaving corrected data for {variable} to database...")
    
    df_save = df.dropna(subset=['AAI_Corrected']).copy()
    
    cols_to_drop = [col for col in df_save.columns if 'Model_' in col or col in ['Bias']]
    df_save = df_save.drop(columns=cols_to_drop, errors='ignore')
        
    # Mock database saving
    print(f"Mocked saving of table '{variable}_AAI_Corrected' complete. Data Head:\n")
    print(df_save.head())
    print(f"\nTable contains {len(df_save)} corrected entries.")



import sys
import os

class Tee:
    def __init__(self, filepath):
        self.file = open(filepath, "w")
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

def main():
    """Main execution block."""

    # =============================
    # LOG FILE SETUP
    # =============================

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log_file = os.path.join(OUTPUT_DIR, "AAI_Run_Log.txt")

    # Enable dual printing (console + file)
    sys.stdout = Tee(log_file)

    for variable in VAR_NAMES:
        print(f"================ STARTING VARIABLE: {variable} ================")
        
        df = load_data(variable, Data_Type)

        df = df[df['Step'] <= 6]
        df = df[(df['Obs'] < 60) & (df['Obs'] > -10)]
        df = df[(df['S2S'] < 60) & (df['S2S'] > -10)]

        df['Bias'] = df['Obs'] - df['S2S']

        df = df.sort_values(by=['Year','Month','Step']).reset_index(drop=True)

        print("\nAfter cleaning:")
        print(df.describe().loc[['min', 'max']])

        if len(df) < WINDOW_SIZE + TEST_WINDOW:
            print(f"Dataset too small ({len(df)} rows). Skipping.")
            continue

        df_train = df[df['Year'] <= 2019].copy()

        X_train, y_train, S2S_train, obs_train = prepare_data(df_train)

        best_model_name, best_model_template, initial_scaler = select_best_model(
            X_train, y_train, S2S_train, obs_train
        )

        print(f"\nBest model selected: {best_model_name}")

        df_adapt = df[df['Year'] >= 2020].copy()

        final_df = run_adaptive_correction(
            df_adapt,
            best_model_template,
            best_model_name,
            initial_scaler
        )

        save_results(final_df, variable)

        print(f"================ ENDING VARIABLE: {variable} ================\n")

    print(f"\n✅ Logs also saved to file.")

if __name__ == "__main__":
    main()
