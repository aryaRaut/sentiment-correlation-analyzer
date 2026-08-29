"""
Ensemble Machine Learning & Deep Learning Module for Sentiment-Correlation Analyzer.

Includes:
1. Soft Voting Ensemble (XGBoost, Random Forest, Logistic Regression)
2. Multi-Horizon Prediction (T+1, T+3, T+5, T+10 horizons)
3. PyTorch LSTM Neural Network sequence classifier with train/val loss curves.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.prediction_model import SentimentPredictor
from src.utils import setup_logger

logger = setup_logger("ensemble_model")

# Standard feature set
FEATURE_COLS = SentimentPredictor.FEATURE_COLS

def create_ensemble(X_train: pd.DataFrame, y_train: pd.Series) -> VotingClassifier:
    """
    Create a soft voting ensemble of XGBoost, Random Forest, and Logistic Regression.
    
    Args:
        X_train (pd.DataFrame): Feature training data.
        y_train (pd.Series): Target labels.
        
    Returns:
        VotingClassifier: Trained soft-voting ensemble model.
    """
    logger.info("Building and training Voting Ensemble...")
    xgb_clf = XGBClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric="logloss"
    )
    rf_clf = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42
    )
    lr_clf = LogisticRegression(
        max_iter=1000, random_state=42
    )

    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb_clf),
            ('rf', rf_clf),
            ('lr', lr_clf)
        ],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)
    return ensemble

def evaluate_ensemble_models(df: pd.DataFrame) -> tuple:
    """
    Evaluate individual models (XGBoost, Random Forest, Logistic Regression) vs Ensemble.
    
    Returns:
        tuple: (comparison_df, ensemble_model)
    """
    sp = SentimentPredictor(df)
    X_train, X_test, y_train, y_test, _ = sp.time_based_split()

    xgb_clf = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, eval_metric="logloss").fit(X_train, y_train)
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_train, y_train)
    lr_clf = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)
    ensemble_clf = create_ensemble(X_train, y_train)

    models = {
        "XGBoost Classifier": xgb_clf,
        "Random Forest Classifier": rf_clf,
        "Logistic Regression": lr_clf,
        "Ensemble (Voting Classifier)": ensemble_clf
    }

    results = []
    for name, model in models.items():
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1 Score": round(f1, 4)
        })

    comp_df = pd.DataFrame(results)
    return comp_df, ensemble_clf

def train_multi_horizon_models(df: pd.DataFrame, horizons: list = [1, 3, 5, 10]) -> dict:
    """
    Train separate XGBoost models for multiple prediction horizons.
    
    Args:
        df (pd.DataFrame): Processed dataframe.
        horizons (list): List of horizon days [1, 3, 5, 10].
        
    Returns:
        dict: Models per horizon and evaluation metrics dataframe.
    """
    logger.info(f"Training Multi-Horizon Models for horizons: {horizons}")
    df_copy = df.copy().sort_values(["Symbol", "Date"] if "Symbol" in df.columns else ["stock", "Date"]).reset_index(drop=True)
    stock_col = "Symbol" if "Symbol" in df_copy.columns else "stock"
    close_col = "Close"

    models_dict = {}
    horizon_results = []

    for h in horizons:
        # Create multi-day target
        df_copy[f"return_{h}d"] = df_copy.groupby(stock_col)[close_col].pct_change(periods=h).shift(-h)
        df_copy[f"target_{h}d"] = df_copy[f"return_{h}d"].apply(lambda x: 1 if x > 0 else (0 if not pd.isna(x) else np.nan))

        valid_df = df_copy.dropna(subset=[f"target_{h}d"] + FEATURE_COLS).copy()
        if len(valid_df) < 50:
            continue

        split_idx = int(len(valid_df) * 0.8)
        train_df = valid_df.iloc[:split_idx]
        test_df = valid_df.iloc[split_idx:]

        X_tr, y_tr = train_df[FEATURE_COLS], train_df[f"target_{h}d"]
        X_te, y_te = test_df[FEATURE_COLS], test_df[f"target_{h}d"]

        model = XGBClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric="logloss"
        )
        model.fit(X_tr, y_tr)
        models_dict[f"{h}D"] = model

        preds = model.predict(X_te)
        acc = accuracy_score(y_te, preds)
        prec = precision_score(y_te, preds, zero_division=0)
        rec = recall_score(y_te, preds, zero_division=0)
        f1 = f1_score(y_te, preds, zero_division=0)

        horizon_results.append({
            "Horizon": f"{h}-Day Forecast (T+{h})",
            "Horizon_Days": h,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1 Score": round(f1, 4),
            "Test Samples": len(y_te)
        })

    results_df = pd.DataFrame(horizon_results)
    return {
        "models": models_dict,
        "results_df": results_df
    }

# -------------------------------------------------------------
# PyTorch LSTM Module Implementation
# -------------------------------------------------------------
class PyTorchLSTMModel(nn.Module):
    """LSTM Neural Network architecture for sentiment-driven sequence classification."""
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 2, dropout: float = 0.2):
        super(PyTorchLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc1 = nn.Linear(hidden_dim, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take last time step
        out = self.relu(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        return out

def build_lstm_sequences(df: pd.DataFrame, window_size: int = 5) -> tuple:
    """Creates sequence datasets [samples, window_size, features] for LSTM."""
    target_col = "actual_direction" if "actual_direction" in df.columns else "target_up"
    stock_col = "Symbol" if "Symbol" in df.columns else "stock"

    # Fill feature missing values
    df_clean = df.copy().sort_values([stock_col, "Date"]).reset_index(drop=True)
    for col in FEATURE_COLS:
        if col not in df_clean.columns:
            df_clean[col] = 0.0
        df_clean[col] = df_clean[col].fillna(0.0)

    X_seqs, y_seqs = [], []

    for _, group in df_clean.groupby(stock_col):
        group_feat = group[FEATURE_COLS].values
        group_target = group[target_col].values
        
        # Normalize features per stock for stability
        std = np.std(group_feat, axis=0)
        std[std == 0] = 1.0
        group_feat_norm = (group_feat - np.mean(group_feat, axis=0)) / std

        for i in range(len(group) - window_size):
            X_seqs.append(group_feat_norm[i : i + window_size])
            y_seqs.append(group_target[i + window_size])

    if len(X_seqs) == 0:
        return np.array([]), np.array([])

    return np.array(X_seqs, dtype=np.float32), np.array(y_seqs, dtype=np.float32)

def train_lstm_model(df: pd.DataFrame, epochs: int = 25, window_size: int = 5, lr: float = 0.005) -> dict:
    """
    Builds, trains, and evaluates PyTorch LSTM model.
    
    Returns:
        dict: model, metrics, loss history curves.
    """
    logger.info("Training PyTorch LSTM Neural Network...")
    X_all, y_all = build_lstm_sequences(df, window_size=window_size)

    if len(X_all) < 30:
        return {
            "accuracy": 0.5,
            "f1_score": 0.5,
            "train_losses": [],
            "val_losses": [],
            "error": "Insufficient sequence data for LSTM"
        }

    # Time-based split
    split_idx = int(len(X_all) * 0.8)
    X_train, y_train = torch.tensor(X_all[:split_idx]), torch.tensor(y_all[:split_idx]).unsqueeze(1)
    X_val, y_val = torch.tensor(X_all[split_idx:]), torch.tensor(y_all[split_idx:]).unsqueeze(1)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    input_dim = X_all.shape[2]
    model = PyTorchLSTMModel(input_dim=input_dim, hidden_dim=32, num_layers=2)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * len(batch_x)

        epoch_train_loss /= len(train_dataset)
        train_losses.append(round(epoch_train_loss, 4))

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
            val_losses.append(round(val_loss, 4))

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        preds_prob = model(X_val).numpy().flatten()
        preds_class = (preds_prob >= 0.5).astype(int)
        y_true = y_val.numpy().flatten().astype(int)

    acc = accuracy_score(y_true, preds_class)
    f1 = f1_score(y_true, preds_class, zero_division=0)
    prec = precision_score(y_true, preds_class, zero_division=0)
    rec = recall_score(y_true, preds_class, zero_division=0)

    return {
        "model": model,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "epochs": epochs,
        "history_df": pd.DataFrame({
            "Epoch": list(range(1, epochs + 1)),
            "Training Loss": train_losses,
            "Validation Loss": val_losses
        })
    }

if __name__ == "__main__":
    from src.data_loader import load_processed_data
    df = load_processed_data()
    comp_df, _ = evaluate_ensemble_models(df)
    print("Ensemble Models Comparison:")
    print(comp_df)
    multi_res = train_multi_horizon_models(df)
    print("\nMulti-Horizon Results:")
    print(multi_res["results_df"])
    lstm_res = train_lstm_model(df, epochs=15)
    print("\nLSTM Accuracy:", lstm_res["accuracy"], "F1:", lstm_res["f1_score"])
