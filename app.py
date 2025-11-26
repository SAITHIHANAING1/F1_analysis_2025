import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Page config
st.set_page_config(
    page_title="Singapore GP Predictor",
    page_icon="🏁",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for minimalist design
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1f1f1f;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        height: 3rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-title">🏁 Singapore GP Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict Formula 1 Singapore Grand Prix winners using machine learning</p>', unsafe_allow_html=True)

# Load and cache data
@st.cache_data
def load_training_data():
    """Load the properly structured training data"""
    df = pd.read_csv("data/training_dataset.csv")
    return df

@st.cache_data
def get_latest_drivers():
    """Get drivers from the most recent race for prediction"""
    df = pd.read_csv("data/training_dataset.csv")
    latest_year = df['year'].max()
    latest_race = df[df['year'] == latest_year].copy()
    return latest_race, latest_year

@st.cache_resource
def train_model(model_type, params, X, y):
    """Train model with caching for efficiency"""
    if model_type == "Random Forest":
        model = RandomForestClassifier(**params, random_state=42)
    elif model_type == "Logistic Regression":
        model = LogisticRegression(**params, max_iter=1000, random_state=42)
    elif model_type == "XGBoost":
        model = XGBClassifier(**params, eval_metric="logloss", random_state=42)
    elif model_type == "SVM":
        model = SVC(**params, probability=True, random_state=42)
    
    model.fit(X, y)
    return model

try:
    training_data = load_training_data()
    latest_race, latest_year = get_latest_drivers()
    
    # Define feature columns
    feature_cols = [
        'driver_sg_races', 'driver_sg_wins', 'driver_sg_avg_finish',
        'driver_sg_best_finish', 'driver_sg_podiums', 'driver_sg_recent_avg',
        'driver_sg_recent_podiums', 'team_sg_races', 'team_sg_wins',
        'team_sg_winrate', 'team_sg_avg_finish', 'team_sg_podiums'
    ]
    
    # Sidebar - Model Selection
    with st.sidebar:
        st.header("Model Settings")
        
        model_type = st.selectbox(
            "Algorithm",
            ["XGBoost", "Random Forest", "Logistic Regression", "SVM", "Ensemble"],
            help="Choose the machine learning algorithm"
        )
        
        st.divider()
        
        # Model parameters based on selection
        params = {}
        if model_type == "Random Forest":
            params['n_estimators'] = st.slider("Trees", 50, 300, 100, 50)
            params['max_depth'] = st.slider("Max Depth", 3, 15, 10)
        elif model_type == "Logistic Regression":
            params['C'] = st.slider("Regularization", 0.1, 5.0, 1.0, 0.1)
        elif model_type == "XGBoost":
            params['n_estimators'] = st.slider("Trees", 50, 300, 100, 50)
            params['learning_rate'] = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)
            params['max_depth'] = st.slider("Max Depth", 3, 10, 6)
        elif model_type == "SVM":
            params['C'] = st.slider("Regularization", 0.1, 5.0, 1.0, 0.1)
            params['kernel'] = st.selectbox("Kernel", ["rbf", "linear"])
        
        st.divider()
        
        # Training data info
        st.caption(f"📊 Training Data")
        st.caption(f"Years: {training_data['year'].min()}-{training_data['year'].max()}")
        st.caption(f"Races: {training_data['year'].nunique()}")
        st.caption(f"Samples: {len(training_data)}")
        
        st.divider()
        train_button = st.button("Train Model", type="primary", use_container_width=True)
    
    # Prepare data for training
    X = training_data[feature_cols]
    y = training_data['win']
    
    # Features for latest race prediction
    X_latest = latest_race[feature_cols]
    
    # Training logic
    if train_button:
        with st.spinner("Training model..."):
            if model_type == "Ensemble":
                models = [
                    LogisticRegression(max_iter=1000, random_state=42),
                    RandomForestClassifier(n_estimators=100, random_state=42),
                    XGBClassifier(eval_metric="logloss", random_state=42),
                    SVC(probability=True, random_state=42)
                ]
                latest_predictions = []
                for model in models:
                    model.fit(X, y)
                    latest_predictions.append(model.predict_proba(X_latest)[:, 1])
                latest_probs = np.mean(latest_predictions, axis=0)
            else:
                model = train_model(model_type, params, X, y)
                latest_probs = model.predict_proba(X_latest)[:, 1]
            
            st.session_state['trained'] = True
            st.session_state['predictions'] = latest_probs
            st.session_state['model_type'] = model_type
            st.session_state['model'] = model if model_type != "Ensemble" else models
            st.session_state['params'] = params
            st.success("Model trained successfully!")
            st.rerun()
    
    # Main content
    if st.session_state.get('trained', False):
        # Create results dataframe for latest race
        results = latest_race[['driver_name', 'team_name']].copy()
        results['win_probability'] = st.session_state['predictions'] * 100
        results = results.sort_values('win_probability', ascending=False).reset_index(drop=True)
        
        # Top prediction highlight
        winner = results.iloc[0]
        st.markdown(f"### 🏆 Predicted Winner - {latest_year} Singapore GP")
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.metric("Driver", winner['driver_name'])
        with col2:
            st.metric("Team", winner['team_name'])
        with col3:
            st.metric("Probability", f"{winner['win_probability']:.1f}%")
        
        st.caption(f"Model: {st.session_state.get('model_type', 'Unknown')}")
        
        st.divider()
        
        # Top 10 contenders
        st.markdown("### Top 10 Contenders")
        top_10 = results.head(10)
        
        for idx, row in top_10.iterrows():
            col1, col2, col3 = st.columns([3, 3, 2])
            with col1:
                st.write(f"**{idx + 1}. {row['driver_name']}**")
            with col2:
                st.write(row['team_name'])
            with col3:
                st.progress(min(row['win_probability'] / 100, 1.0))
                st.caption(f"{row['win_probability']:.1f}%")
        
        st.divider()
        
        # Download option
        csv = results.to_csv(index=False)
        st.download_button(
            "📥 Download All Predictions",
            csv,
            f"singapore_gp_{latest_year}_predictions.csv",
            "text/csv",
            use_container_width=True
        )
        
        # Historical validation section
        st.markdown("### 📈 Model Validation")
        st.caption("How well did this model predict previous races?")
        
        with st.expander("View Past Years Performance", expanded=False):
            # Get last 3 years for validation
            years_to_show = sorted(training_data['year'].unique())[-3:]
            model_type_used = st.session_state.get('model_type')
            params_used = st.session_state.get('params', {})
            
            for year in years_to_show:
                year_data = training_data[training_data['year'] == year].copy()
                X_year = year_data[feature_cols]
                
                # Train on data before this year
                train_mask = training_data['year'] < year
                X_train_temp = training_data[train_mask][feature_cols]
                y_train_temp = training_data[train_mask]['win']
                
                if len(X_train_temp) > 10:
                    if model_type_used == "Ensemble":
                        temp_models = [
                            LogisticRegression(max_iter=1000, random_state=42),
                            RandomForestClassifier(n_estimators=100, random_state=42),
                            XGBClassifier(eval_metric="logloss", random_state=42),
                            SVC(probability=True, random_state=42)
                        ]
                        temp_preds = []
                        for m in temp_models:
                            m.fit(X_train_temp, y_train_temp)
                            temp_preds.append(m.predict_proba(X_year)[:, 1])
                        year_probs = np.mean(temp_preds, axis=0)
                    else:
                        temp_model = train_model(model_type_used, params_used, X_train_temp, y_train_temp)
                        year_probs = temp_model.predict_proba(X_year)[:, 1]
                    
                    year_data['predicted_prob'] = year_probs
                    top_3 = year_data.sort_values('predicted_prob', ascending=False).head(3)
                    
                    st.markdown(f"**{year}:**")
                    for i, (_, row) in enumerate(top_3.iterrows(), 1):
                        actual = " ✅ **ACTUAL WINNER**" if row['win'] == 1 else ""
                        st.write(f"{i}. {row['driver_name']} - {row['predicted_prob']*100:.1f}%{actual}")
                    st.write("")
    
    else:
        # Initial state - show instructions
        st.info(f"👈 Configure your model settings and click **Train Model** to predict the {latest_year} Singapore GP winner")
        
        # Show dataset stats
        st.markdown("### 📊 Training Dataset Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Races", training_data['year'].nunique())
        with col2:
            st.metric("Years", f"{training_data['year'].min()}-{training_data['year'].max()}")
        with col3:
            st.metric("Drivers", training_data['driver_name'].nunique())
        
        st.markdown("### 🏁 Recent Winners")
        recent_winners = training_data[training_data['win'] == 1].sort_values('year', ascending=False).head(5)
        for _, row in recent_winners.iterrows():
            st.write(f"**{int(row['year'])}:** {row['driver_name']} ({row['team_name']})")

except FileNotFoundError:
    st.error("⚠️ Training data not found")
    st.info("Please run these scripts first:")
    st.code("""
python load_data.py
python merge_data.py
python create_training_data.py
    """, language="bash")

# Footer
st.divider()
st.caption("Data: Kaggle Formula 1 Championship Dataset (1950-2020)")
