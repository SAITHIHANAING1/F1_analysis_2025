import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.svm import SVC

# Load training data
df = pd.read_csv("data/training_dataset.csv")

# Prepare features and target
feature_cols = [
    'driver_sg_races', 'driver_sg_wins', 'driver_sg_avg_finish',
    'driver_sg_best_finish', 'driver_sg_podiums', 'driver_sg_recent_avg',
    'driver_sg_recent_podiums', 'team_sg_races', 'team_sg_wins',
    'team_sg_winrate', 'team_sg_avg_finish', 'team_sg_podiums'
]

X = df[feature_cols]
y = df['win']

# Split data - use last 2 races as test set
train_years = df['year'].unique()[:-2]
test_years = df['year'].unique()[-2:]

train_mask = df['year'].isin(train_years)
test_mask = df['year'].isin(test_years)

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
print(f"Training years: {train_years.min()}-{train_years.max()}")
print(f"Test years: {list(test_years)}")

# Train models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric='logloss', random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42)
}

results = []

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    if y_test.sum() > 0:  # Only calculate AUC if there are positive samples
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    else:
        roc_auc = None
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'ROC-AUC': roc_auc
    })
    
    print(f"Accuracy: {accuracy:.3f}")
    if roc_auc:
        print(f"ROC-AUC: {roc_auc:.3f}")
    
    # Show predictions for test years
    test_df = df[test_mask].copy()
    test_df['predicted_prob'] = y_pred_proba
    
    for year in test_years:
        year_data = test_df[test_df['year'] == year].sort_values('predicted_prob', ascending=False)
        print(f"\n{year} Singapore GP - Top 5 Predictions:")
        for i, row in year_data.head(5).iterrows():
            actual = "✓ WON" if row['win'] == 1 else ""
            print(f"  {row['driver_name']:25s} ({row['team_name']:20s}) - {row['predicted_prob']*100:5.1f}% {actual}")

# Save results
results_df = pd.DataFrame(results)
print("\n" + "="*50)
print("\nModel Comparison:")
print(results_df.to_string(index=False))
results_df.to_csv("data/model_evaluation.csv", index=False)
