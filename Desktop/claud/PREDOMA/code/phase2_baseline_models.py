#!/usr/bin/env python3
"""
PREDOMA Phase 2: Baseline Machine Learning Models
Trains and evaluates baseline models for mule account detection.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'config'))
import phase1_config as config

# ============================================================================
# LOGGING & SETUP
# ============================================================================

def log_message(msg, tag="PHASE2"):
    """Log timestamped message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{tag:<12}] {msg}")

def setup_directories():
    """Create necessary output directories."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    log_message("Directories verified", "SETUP")

# ============================================================================
# LOAD & PREPARE DATA
# ============================================================================

def load_features_matrix():
    """Load the Phase 1 features matrix."""
    features_file = config.OUTPUT_FILES['features_matrix']

    if not os.path.exists(features_file):
        log_message(f"ERROR: Features file not found: {features_file}", "ERROR")
        return None

    df = pd.read_csv(features_file)
    log_message(f"Loaded {len(df)} labeled accounts with {len(df.columns)} columns", "DATA_LOAD")

    return df

def prepare_training_data(df):
    """Prepare features and labels for training."""
    # Separate features and labels
    X = df.drop(['account_id', 'label'], axis=1)
    y = df['label']

    # Handle missing values
    X = X.fillna(X.mean())

    log_message(f"Features shape: {X.shape}", "DATA_PREP")
    log_message(f"Label distribution: {y.value_counts().to_dict()}", "DATA_PREP")

    return X, y

# ============================================================================
# TRAIN & EVALUATE MODELS
# ============================================================================

def train_evaluate_model(model_name, model, X_train, X_test, y_train, y_test):
    """Train and evaluate a single model."""

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    # Evaluate
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    cm = confusion_matrix(y_test, y_pred)

    log_message(f"{model_name} trained successfully", "MODEL")
    log_message(f"  Accuracy:  {acc:.4f}", "EVAL")
    log_message(f"  Precision: {prec:.4f}", "EVAL")
    log_message(f"  Recall:    {rec:.4f}", "EVAL")
    log_message(f"  F1-Score:  {f1:.4f}", "EVAL")
    if auc is not None:
        log_message(f"  ROC-AUC:   {auc:.4f}", "EVAL")
    log_message(f"  Confusion Matrix:\n    {cm}", "EVAL")

    return {
        'model_name': model_name,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'roc_auc': auc,
        'confusion_matrix': cm,
        'model': model
    }

def run_baseline_models(X, y):
    """Train and evaluate all baseline models."""

    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_SEED, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_message(f"Train set: {len(X_train)} samples", "SPLIT")
    log_message(f"Test set: {len(X_test)} samples", "SPLIT")

    # Initialize models
    models = [
        ("Logistic Regression", LogisticRegression(random_state=config.RANDOM_SEED, max_iter=1000)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_SEED, n_jobs=-1)),
    ]

    if HAS_XGBOOST:
        models.append(("XGBoost", xgb.XGBClassifier(random_state=config.RANDOM_SEED, use_label_encoder=False, eval_metric='logloss')))

    # Train and evaluate
    results = []
    for model_name, model in models:
        # Use scaled data for LR, original for tree models
        if "Logistic" in model_name:
            result = train_evaluate_model(model_name, model, X_train_scaled, X_test_scaled, y_train, y_test)
        else:
            result = train_evaluate_model(model_name, model, X_train, X_test, y_train, y_test)
        results.append(result)

    return results

# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(results):
    """Save model evaluation results."""

    # Create results summary
    summary = []
    for result in results:
        summary.append({
            'Model': result['model_name'],
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1-Score': result['f1_score'],
            'ROC-AUC': result['roc_auc']
        })

    summary_df = pd.DataFrame(summary)
    summary_file = os.path.join(config.RESULTS_DIR, 'phase2_model_comparison.csv')
    summary_df.to_csv(summary_file, index=False)

    log_message(f"Saved: {summary_file}", "SAVE")

    # Print summary
    print("\n" + "="*70)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*70)
    print(summary_df.to_string(index=False))
    print("="*70 + "\n")

    return summary_df

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main Phase 2 execution."""
    print("\n" + "="*70)
    print("PREDOMA PHASE 2: BASELINE MODELS")
    print("="*70 + "\n")

    setup_directories()

    # Load Phase 1 results
    df = load_features_matrix()
    if df is None:
        return False

    # Prepare data
    X, y = prepare_training_data(df)

    # Train baseline models
    log_message("Training baseline models...", "MODELS")
    results = run_baseline_models(X, y)

    # Save results
    summary = save_results(results)

    # Summary
    best_model_idx = summary['F1-Score'].idxmax()
    best_model_name = summary.loc[best_model_idx, 'Model']
    best_f1 = summary.loc[best_model_idx, 'F1-Score']

    log_message(f"Best model: {best_model_name} (F1={best_f1:.4f})", "COMPLETE")

    print("="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Review model comparison in: ./results/phase2_model_comparison.csv")
    print("2. Prepare research paper with findings")
    print("3. Push to git repository")
    print("="*70 + "\n")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
