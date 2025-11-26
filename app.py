import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

st.set_page_config(page_title="F1 Singapore Predictor", layout="wide", initial_sidebar_state="expanded")

with st.sidebar:
    st.write("### Appearance")
    dark_mode = st.toggle("🌙 Dark Mode", value=False)

# 2. DEFINE PALETTES
if dark_mode:
    theme = {
        "bg_main": "#000000",
        "bg_sidebar": "#111111",
        "card_bg": "#1a1a1a",
        "text_primary": "#ffffff",
        "text_secondary": "#a1a1aa",
        "border": "#27272a",
        "shadow": "rgba(255, 255, 255, 0.1)",
        "accent_text": "#ffffff",
        "button_bg": "#ffffff",
        "button_text": "#000000",
        "metric_bg": "#1a1a1a",
        "metric_border": "#27272a"
    }
else:
    theme = {
        "bg_main": "#ffffff",
        "bg_sidebar": "#f4f4f5",
        "card_bg": "#ffffff",
        "text_primary": "#000000",
        "text_secondary": "#52525b",
        "border": "#e4e4e7",
        "shadow": "rgba(0, 0, 0, 0.1)",
        "accent_text": "#000000",
        "button_bg": "#000000",
        "button_text": "#ffffff",
        "metric_bg": "#ffffff",
        "metric_border": "#e4e4e7"
    }

st.markdown(f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    * {{font-family: 'Inter', sans-serif;}}
    
    .stApp, .main {{background-color: {theme['bg_main']};}}
    .main {{padding: 1rem 2rem !important; max-height: 100vh; overflow-y: auto;}}
    
    [data-testid="stSidebar"] {{background-color: {theme['bg_sidebar']}; border-right: 1px solid {theme['border']};}}
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {{color: {theme['text_primary']} !important;}}
    [data-testid="stSidebar"] label {{font-weight: 500;}}
    div[data-testid="stToggle"] label div {{border: 1px solid {theme['text_secondary']} !important;}}
    section[data-testid="stSidebar"] button, [data-testid="collapsedControl"] {{color: {theme['text_primary']} !important;}}
    [data-testid="collapsedControl"] svg, section[data-testid="stSidebar"] button svg {{fill: {theme['text_primary']} !important;}}
    
    /* HEADERS */
    h1 {{
        color: {theme['text_primary']} !important;
        font-size: 1.8rem !important;
        margin-bottom: 0.25rem !important;
        background: transparent;
        text-shadow: none;
    }}
    h2 {{
        color: {theme['text_secondary']} !important;
        font-size: 1rem !important;
        margin-bottom: 1rem !important;
    }}
    h3 {{
        color: {theme['text_primary']} !important;
        font-size: 1.1rem !important;
        margin-top: 1rem !important;
        margin-bottom: 0.5rem !important;
    }}
    
    /* CUSTOM DRIVER CARD */
    .driver-card {{
        background: {theme['card_bg']};
        border-radius: 8px;
        padding: 0.5rem 0.75rem;
        margin-bottom: 0.4rem;
        box-shadow: 0 1px 2px {theme['shadow']};
        border: 1px solid {theme['border']};
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}
    .driver-card:hover {{
        border-color: {theme['accent_text']};
    }}
    
    /* TEXT INFO */
    .driver-name {{
        font-size: 0.9rem;
        font-weight: 600;
        color: {theme['text_primary']};
        margin-bottom: 0.1rem;
    }}
    .team-name {{
        color: {theme['text_secondary']};
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.02em;
    }}
    
    /* PROBABILITY & ACCENTS */
    .prob-value {{
        color: {theme['accent_text']};
        font-size: 1rem;
        font-weight: 600;
    }}
    .prob-bar-bg {{
        background: {theme['border']};
        height: 3px;
        border-radius: 2px;
        margin-top: 0.2rem;
    }}
    .prob-bar-fill {{
        background: {theme['accent_text']};
        height: 100%;
        border-radius: 2px;
    }}
    
    .stButton > button, .stDownloadButton > button {{
        background-color: {theme['button_bg']} !important;
        color: {theme['button_text']} !important;
        border: 2px solid {theme['button_bg']} !important;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        box-shadow: 0 2px 4px {theme['shadow']};
    }}
    .stButton > button *, .stDownloadButton > button * {{
        color: {theme['button_text']} !important;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{opacity: 0.85 !important;}}
    
    /* METRICS */
    [data-testid="stMetric"] {{
        background-color: {theme['metric_bg']};
        border: 1px solid {theme['metric_border']};
        padding: 0.5rem;
        border-radius: 8px;
        box-shadow: 0 1px 2px 0 {theme['shadow']};
    }}
    [data-testid="stMetricLabel"] {{
        color: {theme['text_secondary']} !important;
        font-size: 0.75rem !important;
    }}
    [data-testid="stMetricValue"] {{
        color: {theme['text_primary']} !important;
        font-size: 1.5rem !important;
    }}
    
    /* INPUTS (Selectbox, etc) */
    .stSelectbox > div > div {{
        background-color: {theme['card_bg']};
        color: {theme['text_primary']};
        border: 1px solid {theme['border']};
    }}
    .stSelectbox > div > div svg {{
        fill: {theme['text_primary']} !important;
    }}
    
    /* Slider min/max values */
    .stSlider [data-testid="stTickBarMin"], 
    .stSlider [data-testid="stTickBarMax"] {{
        color: {theme['text_primary']} !important;
    }}
    
    /* Hide Default Elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

</style>""", unsafe_allow_html=True)

st.title("Singapore Grand Prix Predictor")
st.subheader("Machine learning predictions for top 5 finishers")

@st.cache_data
def load_training_data():
    return pd.read_csv("data/training_dataset.csv")

@st.cache_data
def get_latest_drivers():
    df = pd.read_csv("data/training_dataset.csv")
    latest_year = df['year'].max()
    prediction_year = latest_year + 1
    return df[df['year'] == latest_year].copy(), prediction_year

@st.cache_resource
def train_model(model_type, params, X, y):
    if model_type == "Random Forest":
        model = RandomForestClassifier(**params, random_state=42)
    elif model_type == "XGBoost":
        model = XGBClassifier(**params, eval_metric="logloss", random_state=42, verbosity=0)
    elif model_type == "Logistic Regression":
        model = LogisticRegression(**params, random_state=42)
    else:  # SVM
        model = SVC(**params, random_state=42)
    model.fit(X, y)
    return model

try:
    training_data = load_training_data()
    latest_race, latest_year = get_latest_drivers()
    
    feature_cols = ['driver_sg_races', 'driver_sg_wins', 'driver_sg_avg_finish',
                    'driver_sg_best_finish', 'driver_sg_podiums', 'driver_sg_recent_avg',
                    'driver_sg_recent_podiums', 'team_sg_races', 'team_sg_wins',
                    'team_sg_winrate', 'team_sg_avg_finish', 'team_sg_podiums']
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("### Model Configuration")
        
        model_type = st.selectbox("Select Model", ["XGBoost", "Random Forest", "Logistic Regression", "SVM"])
        
        if model_type == "XGBoost":
            learning_rate = st.slider("Learning Rate", 0.05, 0.2, 0.1, 0.01)
            params = {'n_estimators': 100, 'learning_rate': learning_rate, 'max_depth': 6}
        elif model_type == "Random Forest":
            n_trees = st.slider("Number of Trees", 100, 300, 150, 50)
            params = {'n_estimators': n_trees, 'max_depth': 10}
        elif model_type == "Logistic Regression":
            c_value = st.slider("Regularization (C)", 0.01, 10.0, 1.0, 0.1)
            params = {'C': c_value, 'max_iter': 1000}
        else:  # SVM
            c_value = st.slider("C Parameter", 0.1, 10.0, 1.0, 0.1)
            params = {'C': c_value, 'kernel': 'rbf', 'probability': True}
        
        st.markdown("---")
        train_button = st.button("Predict Results", use_container_width=True)

    X, y = training_data[feature_cols], training_data['win']
    X_latest = latest_race[feature_cols]
    
    if train_button:
        train_model.clear()
        with st.spinner("Analyzing historical data and generating predictions..."):
            model = train_model(model_type, params, X, y)
            latest_probs = model.predict_proba(X_latest)[:, 1]
            
            st.session_state.update({'trained': True, 'predictions': latest_probs,
                                     'model_type': model_type})
            st.rerun()
    
    if st.session_state.get('trained', False):
        results = latest_race[['driver_name', 'team_name']].copy()
        results['win_probability'] = st.session_state['predictions'] * 100
        results = results.sort_values('win_probability', ascending=False).reset_index(drop=True)
        
        st.markdown(f"### Predicted Top 5 - {latest_year} Singapore GP")
        st.caption(f"Based on {st.session_state.get('model_type')} analysis of {training_data['year'].nunique()} years of historical data (2009-2024)")
        
        top_5 = results.head(5)
        max_prob = top_5['win_probability'].max()
        
        for idx, row in top_5.iterrows():
            prob = row['win_probability']
            bar_width = (prob / max_prob) * 100
            
            st.markdown(f"""
            <div class="driver-card">
                <div style="display: flex; align-items: center; flex-grow: 1;">
                    <div class="driver-info">
                        <div class="driver-name">{row['driver_name']}</div>
                        <div class="team-name">{row['team_name']}</div>
                    </div>
                </div>
                <div class="prob-container">
                    <div class="prob-value">{prob:.1f}%</div>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width: {bar_width}%"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Reset Analysis"):
                st.session_state['trained'] = False
                st.rerun()
        with col2:
            st.download_button("Download Full Analysis Report", results.to_csv(index=False),
                               f"singapore_gp_predictions_{latest_year}.csv", "text/csv", use_container_width=True)
        
    else:
        # Stats Overview
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Historical Races", str(training_data['year'].nunique()))
        with c2:
            st.metric("Drivers Analyzed", str(training_data['driver_name'].nunique()))
        with c3:
            st.metric("Teams Analyzed", str(training_data['team_name'].nunique()))
        
        st.markdown("### Recent Singapore GP Winners")
        winners = training_data[training_data['win'] == 1].sort_values('year', ascending=False).head(3)
        
        for _, row in winners.iterrows():
            st.markdown(f"""
            <div style="padding: 0.5rem 0.75rem; background: {theme['card_bg']}; border-radius: 8px; margin-bottom: 0.3rem; border: 1px solid {theme['border']}; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-weight: 600; color: {theme['text_primary']}; font-size: 0.9rem;">{int(row['year'])}</span>
                    <span style="margin: 0 0.75rem; color: {theme['border']};">|</span>
                    <span style="color: {theme['text_primary']}; font-size: 0.9rem;">{row['driver_name']}</span>
                </div>
                <div style="color: {theme['text_secondary']}; font-size: 0.75rem;">{row['team_name']}</div>
            </div>
            """, unsafe_allow_html=True)

except FileNotFoundError:
    st.error("Data not found. Please run the data pipeline scripts first.")

st.markdown("<div style='margin-top: 4rem; text-align: center; color: #999; font-size: 0.8rem;'>F1 Singapore GP Predictor • Powered by Machine Learning</div>", unsafe_allow_html=True)