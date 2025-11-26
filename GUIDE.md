# How to Run the Singapore GP Predictor

## Quick Start

1. **Open terminal in the project directory**

2. **Install required packages** (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure data is prepared** (run once):
   ```bash
   python load_data.py
   python merge_data.py
   python feature_engineering.py
   ```

4. **Launch the web app**:
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** to `http://localhost:8501`

## Using the App

### 1. Select Your Model
- Choose from 5 algorithms: Random Forest, XGBoost, Logistic Regression, SVM, or Ensemble
- Adjust model parameters using the sliders

### 2. Train the Model
- Click "Train Model" button in the sidebar
- Wait for training to complete (~5-10 seconds)

### 3. View Predictions
- See the predicted winner at the top
- Browse top 10 contenders with probability bars
- Download full predictions as CSV

### 4. Custom Analysis (Optional)
- Expand "Create Custom Profile"
- Adjust driver statistics
- Click "Calculate Probability" to see results

## Features

✅ **Minimalist Design** - Clean, focused interface  
✅ **Real-time Training** - Instant model updates  
✅ **Multiple Algorithms** - Compare different ML approaches  
✅ **Custom Predictions** - Test hypothetical scenarios  
✅ **Export Results** - Download predictions as CSV  
✅ **Efficient Caching** - Fast performance with smart data management

## Tips

- Start with **Ensemble** model for best overall accuracy
- Use **Random Forest** for interpretable results
- Try **XGBoost** for highest performance with tuned parameters
- Adjust parameters and retrain to see how they affect predictions
