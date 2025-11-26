# F1 Singapore GP Predictor

Machine learning web app to predict Formula 1 Singapore Grand Prix winners using historical performance data (2009-2024).

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Prepare data (run in order)
python merge_data.py
python create_training_data.py

# Launch web app
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## How It Works

### Proper Time-Series Training

This project uses **proper temporal validation**:
- Each race is predicted using only historical data from *before* that race
- No data leakage - the model never sees future information
- Features are calculated incrementally for each driver/team

### Training Process

1. **Data Preparation**
   - `merge_data.py` - Combines raw F1 datasets and filters Singapore GP races
   - `create_training_data.py` - Creates features from historical data only
   - Builds 237 driver-race combinations across 14 years (2009-2024)

2. **Model Training** (in web app)
   - Trains on all historical data (2009-2024)
   - Predicts 2025 Singapore GP winners
   - Four ML algorithms available: XGBoost, Random Forest, Logistic Regression, SVM

## Dataset

Data sourced from Kaggle's Formula 1 World Championship dataset:
https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020

## Project Structure

```
F1-Analysis/
├── app.py                    # Streamlit web application
├── merge_data.py             # Merge F1 datasets for Singapore GP
├── create_training_data.py   # Generate training features
├── requirements.txt          # Python dependencies
├── data/
│   ├── merged_singapore.csv      # Singapore GP historical data
│   └── training_dataset.csv      # Features for ML training
└── Datasets/                 # Raw CSV files from Kaggle
```

## Features

The model uses 12 key features for prediction:

**Driver Features:**
- Singapore GP races, wins, average finish, best finish, podiums
- Recent form (last 3 races average, recent podiums)

**Team Features:**
- Singapore GP races, wins, win rate, average finish, podiums

## Models

Four machine learning algorithms available:
1. **XGBoost** - Gradient boosting (default)
2. **Random Forest** - Ensemble decision trees
3. **Logistic Regression** - Linear classification
4. **SVM** - Support Vector Machine with RBF kernel

## Web Application Features

- 🎨 Dark/Light theme toggle
- ⚙️ Real-time model parameter tuning
- 📊 Top 5 predictions with probability bars
- 📈 Historical race statistics
- 💾 Download full predictions as CSV
- 🎯 Compact single-page layout

## Requirements

- pandas
- scikit-learn
- xgboost
- streamlit
- numpy
