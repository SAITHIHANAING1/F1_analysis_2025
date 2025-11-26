# How to Run the Singapore GP Predictor

## Quick Start

1. **Open terminal in the project directory**

2. **Install required packages** (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare data** (run once or when updating dataset):
   ```bash
   python merge_data.py
   python create_training_data.py
   ```

4. **Launch the web app**:
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** to `http://localhost:8501`

## Using the App

### 1. Theme Selection
- Toggle between Dark Mode and Light Mode using the 🌙 button in the sidebar
- Clean black and white design for optimal readability

### 2. Select Your Model
- Choose from 4 algorithms: XGBoost, Random Forest, Logistic Regression, or SVM
- Adjust model parameters using the sliders:
  - **XGBoost**: Learning Rate (0.05-0.2)
  - **Random Forest**: Number of Trees (100-300)
  - **Logistic Regression**: Regularization C (0.01-10)
  - **SVM**: C Parameter (0.1-10)

### 3. Generate Predictions
- Click "Predict Results" button in the sidebar
- Wait for training to complete (~2-3 seconds)

### 4. View Results
- See top 5 predicted drivers with win probabilities
- View recent Singapore GP winners (2022-2024)
- Check historical statistics: races analyzed, drivers, teams
- Download full analysis report as CSV

## Data Pipeline

### Step 1: Merge Data (`merge_data.py`)
- Combines races, results, drivers, constructors, and circuits datasets
- Filters for Singapore GP races only
- Creates `data/merged_singapore.csv` with 318 rows

### Step 2: Create Training Data (`create_training_data.py`)
- Generates features from historical Singapore GP data
- Calculates driver and team statistics
- Creates `data/training_dataset.csv` with 237 samples (2009-2024)

### Step 3: Run Web App (`app.py`)
- Trains selected ML model on historical data
- Predicts 2025 Singapore GP top 5 finishers
- Interactive interface with theme customization

## Features

✅ **Dark/Light Theme** - Toggle between modes  
✅ **4 ML Algorithms** - XGBoost, Random Forest, Logistic Regression, SVM  
✅ **Real-time Training** - Instant model updates  
✅ **Parameter Tuning** - Adjust model settings interactively  
✅ **Top 5 Predictions** - View most likely winners  
✅ **Export Results** - Download predictions as CSV  
✅ **Compact Layout** - Single-page design, no scrolling  
✅ **Efficient Caching** - Fast performance with smart data management

## Tips

- Start with **XGBoost** for best accuracy
- Use **Random Forest** for stable, interpretable results
- Try **Logistic Regression** for fast training
- Adjust parameters and retrain to see impact on predictions
- Download CSV for detailed analysis of all drivers
