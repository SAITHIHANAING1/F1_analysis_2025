import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict

features = pd.read_csv("data/feature_dataset.csv")
features['ever_win'] = (features['sg_win_count'] > 0).astype(int)

X = features.drop(columns=['driver_name', 'team_name', 'ever_win'])
y = features['ever_win']

lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

lr_probs = cross_val_predict(lr, X, y, cv=5, method='predict_proba')[:, 1]
rf_probs = cross_val_predict(rf, X, y, cv=5, method='predict_proba')[:, 1]

output = pd.DataFrame({
    'driver_name': features['driver_name'],
    'team_name': features['team_name'],
    'lr_prob': lr_probs,
    'rf_prob': rf_probs
})

output.to_csv("data/predictions_lr_rf.csv", index=False)
print(f"LR & RF predictions: {len(output)} drivers")
