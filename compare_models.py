import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

# Load and merge predictions
lr_rf = pd.read_csv("data/predictions_lr_rf.csv")
xgb_svm = pd.read_csv("data/predictions_xgb_svm.csv")
preds = lr_rf.merge(xgb_svm, on=["driver_name", "team_name"])

# Get true labels
features = pd.read_csv("data/feature_dataset.csv")
features['ever_win'] = (features['sg_win_count'] > 0).astype(int)
y_true = features.set_index("driver_name")["ever_win"].loc[preds["driver_name"]]

# Evaluate models
model_cols = ["lr_prob", "rf_prob", "xgb_prob", "svm_prob"]
results = []

for col in model_cols:
    y_pred = preds[col]
    results.append({
        "model": col.replace('_prob', ''),
        "roc_auc": roc_auc_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, (y_pred > 0.5).astype(int)),
        "log_loss": log_loss(y_true, y_pred)
    })

# Add ensemble
preds["ensemble_prob"] = preds[model_cols].mean(axis=1)
results.append({
    "model": "ensemble",
    "roc_auc": roc_auc_score(y_true, preds["ensemble_prob"]),
    "accuracy": accuracy_score(y_true, (preds["ensemble_prob"] > 0.5).astype(int)),
    "log_loss": log_loss(y_true, preds["ensemble_prob"])
})

# Save results
metrics_df = pd.DataFrame(results)
metrics_df.to_csv("data/model_comparison_metrics.csv", index=False)
preds.to_csv("data/combined_predictions.csv", index=False)

print("\nModel Comparison:")
print(metrics_df.to_string(index=False))
