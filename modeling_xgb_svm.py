import pandas as pd
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict

# Load and prepare data
features = pd.read_csv("data/feature_dataset.csv")
features['ever_win'] = (features['sg_win_count'] > 0).astype(int)

X = features.drop(columns=['driver_name', 'team_name', 'ever_win'])
y = features['ever_win']

# Train models with cross-validation
xgb = XGBClassifier(eval_metric="logloss", random_state=42)
svm = SVC(kernel='rbf', probability=True, random_state=42)

xgb_probs = cross_val_predict(xgb, X, y, cv=5, method='predict_proba')[:, 1]
svm_probs = cross_val_predict(svm, X, y, cv=5, method='predict_proba')[:, 1]

# Save predictions
output = pd.DataFrame({
    'driver_name': features['driver_name'],
    'team_name': features['team_name'],
    'xgb_prob': xgb_probs,
    'svm_prob': svm_probs
})

output.to_csv("data/predictions_xgb_svm.csv", index=False)
print(f"Saved XGBoost & SVM predictions: {len(output)} drivers")
print(output.head())
