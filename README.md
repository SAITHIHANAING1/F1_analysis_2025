# F1 Analysis - Singapore GP Winner Prediction

A machine learning project with an interactive web interface to predict Formula 1 Singapore Grand Prix winners using historical performance data.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Prepare data (run in order)
python load_data.py
python merge_data.py
python create_training_data.py

# Launch web app
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

## How It Works

### Proper Time-Series Training

Unlike typical ML projects, this uses **proper temporal validation**:
- Each race is predicted using only historical data from *before* that race
- No data leakage - the model never sees future information
- Features are calculated incrementally for each driver/team

### Training Process

1. **Data Preparation** (`create_training_data.py`)
   - For each race year, creates features from prior history only
   - Tracks driver performance, team strength, recent form
   - Builds 237 driver-race combinations across 14 years (2009-2024)

2. **Model Training** (in web app)
   - Trains on all historical data
   - Predicts the next Singapore GP winner
   - Validates on past years to show accuracy

## Dataset

Data sourced from Kaggle's Formula 1 World Championship dataset (1950-2020):
https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020

## Project Structure

```
F1-Analysis/
├── load_data.py              # Load raw F1 datasets
├── merge_data.py             # Merge and filter Singapore GP data
├── feature_engineering.py    # Create predictive features
├── modeling_lr_rf.py         # Logistic Regression & Random Forest models
├── modeling_xgb_svm.py       # XGBoost & SVM models
├── compare_models.py         # Compare all model performances
├── data/                     # Generated datasets and predictions
└── Datasets/                 # Raw CSV files from Kaggle
```

## Features

The model uses several key features for prediction:
- **Historical Performance**: Race count, win count, average finish position at Singapore
- **Recent Form**: Last 3 races average finish and podium count
- **Team Strength**: Team win rate at Singapore GP

## Models

Four machine learning models are trained and compared:
1. Logistic Regression
2. Random Forest
3. XGBoost
4. Support Vector Machine (SVM)

An ensemble model combines predictions from all four approaches.

## Usage

### Data Preparation

Run the scripts in order to prepare the data:

```bash
python load_data.py
python merge_data.py
python feature_engineering.py
```

### Model Training (Optional)

Train and compare models using the command line:

```bash
python modeling_lr_rf.py
python modeling_xgb_svm.py
python compare_models.py
```

### Web Application 🚀

Launch the interactive web interface to predict Singapore GP winners:

```bash
streamlit run app.py
```

**Features:**
- Adjust model parameters in real-time
- Compare different ML algorithms (Random Forest, XGBoost, SVM, Logistic Regression, Ensemble)
- View top championship contenders with win probabilities
- Create custom driver profiles and predict their chances
- Download predictions as CSV

## Results

Model performance metrics (ROC-AUC, accuracy, log loss) are saved in `data/model_comparison_metrics.csv`. Individual and combined predictions are available in the `data/` directory.

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Requirements

- pandas
- scikit-learn
- xgboost
- streamlit
- numpy
