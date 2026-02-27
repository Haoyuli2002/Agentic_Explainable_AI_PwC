"""
PatchTST Training Script for Credit Default Risk Dataset
=========================================================
This script trains a PatchTST model from the neuralforecast library.
The key fix for macOS Apple Silicon (M1/M2/M3) bus errors:
  - Disable MPS via environment variables BEFORE importing torch
  - Use torch.multiprocessing.set_start_method('spawn') for safe process forking
  - Run nf.fit() and nf.predict() separately instead of nf.cross_validation()
"""

import os

# === CRITICAL: Must set BEFORE importing torch ===
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Fully disables MPS memory
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import torch.multiprocessing as mp

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def train():
    # Import here so multiprocessing fork is safe
    from neuralforecast.core import NeuralForecast
    from neuralforecast.losses.pytorch import DistributionLoss
    from neuralforecast.losses.pytorch import Bernoulli
    from neuralforecast.models import PatchTST

    # -------------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------------
    print("=" * 50)
    print("Step 1: Loading data...")
    data_path = os.path.join("datasets", "Credit Default Prediction Dataset",
                             "credit_risk_dataset_5k.csv")

    if not os.path.exists(data_path):
        print(f"ERROR: Dataset not found at '{data_path}'")
        print("Please make sure you run this script from the root of the project.")
        return

    df = pd.read_csv(data_path, sep=";", decimal=",")
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns.")

    # -------------------------------------------------------
    # 2. Preprocess
    # -------------------------------------------------------
    print("Step 2: Preprocessing data...")
    df.replace(-9999.0, np.nan, inplace=True)
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)

    # -------------------------------------------------------
    # 3. Format for NeuralForecast (unique_id, ds, y)
    # -------------------------------------------------------
    print("Step 3: Formatting for NeuralForecast...")
    df["unique_id"] = df["id"].astype(str)
    df["ds"] = pd.to_datetime(df["time_series"].astype(str), format="%Y%m")
    df["y"] = df["Default Flag"].astype(float)

    df = df.sort_values(by=["unique_id", "ds"]).reset_index(drop=True)
    df_final = df[["unique_id", "ds", "y"]].copy()

    # Split: use all but last step per series for training
    max_ds_per_id = df_final.groupby("unique_id")["ds"].max()
    df_train = df_final[~df_final.set_index(["unique_id", "ds"]).index.isin(
        [(uid, ds) for uid, ds in max_ds_per_id.items()]
    )].copy()
    df_test = df_final[df_final.set_index(["unique_id", "ds"]).index.isin(
        [(uid, ds) for uid, ds in max_ds_per_id.items()]
    )].copy()

    print(f"  Train size: {len(df_train)}, Test size for prediction: {len(df_test)}")

    # -------------------------------------------------------
    # 4. Initialize PatchTST Model
    # -------------------------------------------------------
    print("Step 4: Initializing PatchTST Model...")
    horizon    = 1  # Predict 1 step ahead
    lookback   = 8  # 8 steps back; 63% of customers have >= 9 data points

    # Bernoulli loss = sigmoid + Binary Cross-Entropy
    # -> directly models P(default=1), ideal for binary classification
    model = PatchTST(
        h=horizon,
        input_size=lookback,
        patch_len=2,
        stride=1,
        revin=True,
        hidden_size=16,
        n_heads=4,
        scaler_type="standard",
        loss=DistributionLoss(distribution='Bernoulli', level=[90]),  # P(default=1)
        learning_rate=1e-3,
        max_steps=2000,
        batch_size=32,
        accelerator="cpu",
        devices=1,
        dataloader_kwargs={"num_workers": 0},
        start_padding_enabled=True,  # Pad short series & mask padded positions
    )

    nf = NeuralForecast(models=[model], freq="YS")

    # -------------------------------------------------------
    # 5. Train
    # -------------------------------------------------------
    print("Step 5: Training model (CPU only)...")
    nf.fit(df=df_train)
    print("  Training complete!")

    # -------------------------------------------------------
    # 6. Predict
    # -------------------------------------------------------
    print("Step 6: Generating predictions...")
    # Use make_future_dataframe() to get correct future date combinations
    future_df = nf.make_future_dataframe(df=df_train)
    preds = nf.predict(df=df_train)
    print("  Predictions ready!")
    print(f"  Prediction columns: {list(preds.columns)}")  # Debug: show output column names

    # -------------------------------------------------------
    # 7. Evaluate (Binary Classification Metrics)
    # -------------------------------------------------------
    print("Step 7: Evaluating...")
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix, classification_report
    )

    # Normalize dates to year-start for alignment (freq='YS' sets all dates to Jan 1)
    df_test_eval = df_test[["unique_id", "ds", "y"]].copy()
    df_test_eval["ds"] = df_test_eval["ds"].dt.to_period("Y").dt.to_timestamp()

    preds_merged = preds.merge(
        df_test_eval.rename(columns={"y": "y_true"}),
        on=["unique_id", "ds"], how="inner"
    )

    if len(preds_merged) > 0:
        y_true = preds_merged["y_true"].values
        # Use the raw 'PatchTST' column which contains actual mean probabilities
        # (PatchTST-median is the Bernoulli distribution median = 0 when p < 0.5)
        y_prob = preds_merged["PatchTST"].values.clip(0, 1)

        # Diagnostics: show prediction distribution
        print(f"\n  🔍 Prediction Distribution Diagnostics:")
        print(f"     y_prob range: [{y_prob.min():.4f}, {y_prob.max():.4f}]")
        print(f"     y_prob mean : {y_prob.mean():.4f}")
        print(f"     Actual default rate: {y_true.mean():.4f}")

        # Find optimal threshold via F1 sweep
        thresholds = np.arange(y_prob.min(), y_prob.max(), 0.005)
        best_f1, best_threshold = 0, 0.5
        for t in thresholds:
            y_pred_t = (y_prob >= t).astype(int)
            f1_t = f1_score(y_true, y_pred_t, zero_division=0)
            if f1_t > best_f1:
                best_f1 = f1_t
                best_threshold = t

        THRESHOLD = best_threshold
        y_pred = (y_prob >= THRESHOLD).astype(int)

        # Regression metrics (continuous output quality)
        mse = np.mean((y_true - y_prob) ** 2)
        mae = np.mean(np.abs(y_true - y_prob))

        # Classification metrics (binary decision quality)
        accuracy  = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall    = recall_score(y_true, y_pred, zero_division=0)
        f1        = f1_score(y_true, y_pred, zero_division=0)

        # AUC-ROC (requires both classes present)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = float("nan")

        cm = confusion_matrix(y_true, y_pred)

        print(f"\n{'='*50}")
        print(f"  📊 Regression Metrics (Continuous Output)")
        print(f"     MSE  : {mse:.6f}")
        print(f"     MAE  : {mae:.6f}")
        print(f"\n  🎯 Classification Metrics (Optimal Threshold = {THRESHOLD:.4f})")
        print(f"     Accuracy  : {accuracy:.4f}")
        print(f"     Precision : {precision:.4f}")
        print(f"     Recall    : {recall:.4f}")
        print(f"     F1-Score  : {f1:.4f}")
        print(f"     AUC-ROC   : {auc:.4f}")
        print(f"\n  📋 Confusion Matrix:")
        print(f"     TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
        print(f"     FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")
        print(f"\n  📈 Class Distribution in Test Set:")
        print(f"     Non-Default (0): {int((y_true==0).sum())} samples")
        print(f"     Default     (1): {int((y_true==1).sum())} samples")
        print(f"{'='*50}")
    else:
        print("  Could not compute metrics (no overlapping dates after merge).")
        print(preds.head())

    # -------------------------------------------------------
    # 8. Save
    # -------------------------------------------------------
    out_path = "patchtst_predictions.csv"
    preds_merged.to_csv(out_path, index=False) if len(preds_merged) > 0 else preds.to_csv(out_path, index=False)
    print(f"\n  Predictions saved to: {out_path}")
    print("Done!")


if __name__ == "__main__":
    # Use 'spawn' instead of 'fork' to avoid macOS multiprocessing crashes
    mp.set_start_method("spawn", force=True)
    train()
