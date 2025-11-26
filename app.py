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
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        height: 3rem;
        font-weight: 500;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-title">🏁 Singapore GP Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict Formula 1 Singapore Grand Prix winners using machine learning</p>', unsafe_allow_html=True)

# Load and cache data
@st.cache_data
def load_features():
    df = pd.read_csv("data/feature_dataset.csv")
    df['ever_win'] = (df['sg_win_count'] > 0).astype(int)
    return df

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

def train_ensemble(X, y):
    """Train ensemble of all models"""
    models = [
        LogisticRegression(max_iter=1000, random_state=42),
        RandomForestClassifier(n_estimators=100, random_state=42),
        XGBClassifier(eval_metric="logloss", random_state=42),
        SVC(probability=True, random_state=42)
    ]
    
    predictions = []
    for model in models:
        model.fit(X, y)
        predictions.append(model.predict_proba(X)[:, 1])
    
    return np.mean(predictions, axis=0)

try:
    features = load_features()
    
    # Sidebar - Model Selection
    with st.sidebar:
        st.header("Model Settings")
        
        model_type = st.selectbox(
            "Algorithm",
            ["Random Forest", "XGBoost", "Logistic Regression", "SVM", "Ensemble"],
            help="Choose the machine learning algorithm"
        )
        
        st.divider()
        
        # Model parameters based on selection
        params = {}
        if model_type == "Random Forest":
            params['n_estimators'] = st.slider("Trees", 50, 300, 100, 50)
            params['max_depth'] = st.slider("Max Depth", 3, 15, 8)
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
        train_button = st.button("Train Model", type="primary", use_container_width=True)
    
    # Prepare data
    X = features.drop(columns=['driver_name', 'team_name', 'ever_win'])
    y = features['ever_win']
    
    # Training logic
    if train_button:
        with st.spinner("Training model..."):
            if model_type == "Ensemble":
                predictions = train_ensemble(X, y)
            else:
                model = train_model(model_type, params, X, y)
                predictions = model.predict_proba(X)[:, 1]
            
            st.session_state['trained'] = True
            st.session_state['predictions'] = predictions
            st.success("Model trained successfully!")
            st.rerun()
    
    # Main content
    if st.session_state.get('trained', False):
        # Create results dataframe
        results = features[['driver_name', 'team_name']].copy()
        results['win_probability'] = st.session_state['predictions'] * 100
        results = results.sort_values('win_probability', ascending=False).reset_index(drop=True)
        
        # Top prediction highlight
        winner = results.iloc[0]
        st.markdown("### 🏆 Predicted Winner")
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.metric("Driver", winner['driver_name'])
        with col2:
            st.metric("Team", winner['team_name'])
        with col3:
            st.metric("Probability", f"{winner['win_probability']:.1f}%")
        
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
                # Progress bar for probability
                st.progress(row['win_probability'] / 100)
                st.caption(f"{row['win_probability']:.1f}%")
        
        st.divider()
        
        # Download option
        csv = results.to_csv(index=False)
        st.download_button(
            "📥 Download All Predictions",
            csv,
            "singapore_gp_predictions.csv",
            "text/csv",
            use_container_width=True
        )
        
        # Custom prediction section
        st.markdown("### 🎯 Custom Driver Analysis")
        st.caption("Adjust parameters to analyze a custom driver profile")
        
        with st.expander("Create Custom Profile", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                races = st.number_input("Singapore Races", 0, 20, 5, help="Total races at Singapore")
                wins = st.number_input("Singapore Wins", 0, 15, 0, help="Total wins at Singapore")
                avg_finish = st.number_input("Avg Finish", 1.0, 25.0, 10.0, 0.5)
            
            with col2:
                last3_avg = st.number_input("Recent Avg Finish", 1.0, 25.0, 10.0, 0.5)
                last3_podiums = st.number_input("Recent Podiums", 0, 3, 0)
                team_winrate = st.slider("Team Win Rate (%)", 0.0, 50.0, 10.0) / 100
            
            if st.button("Calculate Probability", use_container_width=True):
                custom_input = pd.DataFrame({
                    'sg_race_count': [races],
                    'sg_win_count': [wins],
                    'sg_avg_finish': [avg_finish],
                    'sg_last3_avg_finish': [last3_avg],
                    'sg_last3_podiums': [last3_podiums],
                    'team_sg_winrate': [team_winrate]
                })
                
                if model_type == "Ensemble":
                    probs = []
                    for model_name, model_params in [
                        ("Random Forest", {'n_estimators': 100}),
                        ("Logistic Regression", {'C': 1.0}),
                        ("XGBoost", {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6}),
                        ("SVM", {'C': 1.0, 'kernel': 'rbf'})
                    ]:
                        m = train_model(model_name, model_params, X, y)
                        probs.append(m.predict_proba(custom_input)[:, 1][0])
                    custom_prob = np.mean(probs)
                else:
                    model = train_model(model_type, params, X, y)
                    custom_prob = model.predict_proba(custom_input)[:, 1][0]
                
                st.success(f"### Win Probability: **{custom_prob * 100:.2f}%**")
                
                rank = (results['win_probability'] >= custom_prob * 100).sum() + 1
                st.info(f"This profile would rank **#{rank}** out of {len(results)} drivers")
    
    else:
        # Initial state - show instructions
        st.info("👈 Configure your model settings and click **Train Model** to start")
        
        # Show dataset stats
        st.markdown("### 📊 Dataset Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Drivers", len(features))
        with col2:
            st.metric("Winners", features['ever_win'].sum())
        with col3:
            st.metric("Teams", features['team_name'].nunique())

except FileNotFoundError:
    st.error("⚠️ Data files not found")
    st.info("Please run these scripts first:")
    st.code("""
python load_data.py
python merge_data.py
python feature_engineering.py
    """, language="bash")

# Footer
st.divider()
st.caption("Data: Kaggle Formula 1 Championship Dataset (1950-2020)")

