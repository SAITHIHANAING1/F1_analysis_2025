import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from pathlib import Path

# ---------- 1. CONFIGURATION ----------
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"

st.set_page_config(
    page_title="F1 Singapore Predictor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- 2. THEME ENGINE ----------
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False

dark_mode = st.session_state["dark_mode"]

# Monochrome Theme Palette
theme = {
    "bg_main": "#000000" if dark_mode else "#ffffff",
    "bg_panel": "#111111" if dark_mode else "#f5f5f5",
    "card_bg": "#111111" if dark_mode else "#ffffff",
    "text_primary": "#ffffff" if dark_mode else "#111111",
    "text_secondary": "#cccccc" if dark_mode else "#555555",
    "border": "#333333" if dark_mode else "#e0e0e0",
    "accent": "#ffffff" if dark_mode else "#000000",
    "toggle_track": "#333333" if dark_mode else "#e0e0e0", 
}


# ---------- 3. CSS STYLING ----------
st.markdown(
    f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* 1. Global Font Size Reduction (Safe alternative to Zoom) */
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        font-size: 14px; /* Reduced from default 16px */
    }}

    .stApp {{ background-color: {theme['bg_main']}; }}

    /* 2. Compact Layout Adjustments */
    .block-container {{
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        max-width: 95% !important;
    }}
    .main {{ padding: 0rem 1rem !important; }}
    
    /* Reduce spacing between elements */
    .stMarkdown {{ margin-bottom: -0.5rem; }}
    div[data-testid="column"] {{ gap: 0.5rem; }}

    /* Typography */
    h1 {{ color: {theme['text_primary']}; font-weight: 700; letter-spacing: -0.03em; margin-top: 0 !important; font-size: 1.8rem !important; }}
    h2, h3 {{ color: {theme['text_primary']}; font-weight: 600; font-size: 1.1rem !important; }}
    p, label {{ color: {theme['text_secondary']}; font-size: 0.9rem; }}
    .stCaption {{ margin-bottom: 0px !important; color: {theme['text_secondary']} !important; font-size: 0.8rem; }}

    /* Panels & Cards */
    .settings-panel {{
        background-color: {theme['bg_panel']};
        border-radius: 12px;
        border: 1px solid {theme['border']};
        padding: 1.25rem;
    }}
    .driver-card {{
        background-color: {theme['card_bg']};
        border-radius: 10px;
        border: 1px solid {theme['border']};
        padding: 0.8rem 1rem; /* Compact padding */
        margin-bottom: 0.6rem;
        transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
    }}

    /* Toggle Switch */
    div[data-testid="stToggle"] label div:first-child {{
        border: 1px solid {theme['text_primary']} !important;
        background-color: {theme['toggle_track']} !important; 
        border-radius: 20px !important;
    }}
    div[data-testid="stToggle"] label div:first-child > div {{
        background-color: {theme['text_primary']} !important;
    }}

    /* Buttons & Download Buttons */
    .stButton > button, .stDownloadButton > button {{
        width: 100%;
        border-radius: 8px;
        border: 1px solid {theme['text_primary']}; 
        padding: 0.5rem 1rem; /* Compact padding */
        font-weight: 600;
        font-size: 0.9rem;
        background-color: {theme['text_primary']} !important;
        color: {theme['bg_main']} !important; 
        transition: opacity 0.2s;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        opacity: 0.85;
    }}
    
    .stButton > button *, .stButton > button p, 
    .stDownloadButton > button *, .stDownloadButton > button p {{
        color: {theme['bg_main']} !important;
    }}

    /* Progress Bar */
    .prob-bar-bg {{
        margin-top: 0.4rem;
        height: 5px;
        border-radius: 3px;
        background-color: {theme['bg_panel']};
        overflow: hidden;
    }}
    .prob-bar-fill {{
        height: 100%;
        border-radius: 3px;
        background-color: {theme['accent']};
    }}

    /* Metrics */
    [data-testid="stMetric"] {{
        background-color: {theme['bg_panel']};
        border-radius: 10px;
        border: 1px solid {theme['border']};
        padding: 0.8rem;
    }}
    [data-testid="stMetricLabel"] {{ color: {theme['text_secondary']}; font-size: 0.8rem; }}
    [data-testid="stMetricValue"] {{ color: {theme['text_primary']}; font-size: 1.6rem; }}

    /* Inputs */
    .stSelectbox div[data-baseweb="select"] > div {{
        background-color: {theme['bg_main']};
        color: {theme['text_primary']};
        border-color: {theme['border']};
        border-radius: 8px;
        font-size: 0.9rem;
    }}
    .stSelectbox svg {{ fill: {theme['text_primary']}; }}

    /* Slider Fixes */
    /* Target the label above the slider */
    div[data-testid="stSlider"] label p {{
        font-size: 0.85rem !important;
    }}
    /* Target the thumb value (if visible in this theme) */
    div[data-testid="stSlider"] div[data-testid="stMarkdownContainer"] p {{
        font-size: 0.85rem !important;
    }}

    /* Responsive Adjustments */
    @media (max-width: 768px) {{
        .block-container {{
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }}
        .main {{
            padding: 0 !important;
        }}
    }}

    /* Hide Streamlit Elements */
    #MainMenu, footer, header {{ visibility: hidden; }}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- 4. DATA LOGIC ----------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_DIR / "training_dataset.csv")

@st.cache_data
def get_prediction_context():
    df = pd.read_csv(DATA_DIR / "training_dataset.csv")
    latest_year = int(df["year"].max())
    latest_data = df[df["year"] == latest_year].copy()
    return latest_data, latest_year

@st.cache_resource
def train_model(model_type, X, y, **params):
    if model_type == "Random Forest":
        rf_params = {k: v for k, v in params.items() if k in ['n_estimators', 'max_depth']}
        model = RandomForestClassifier(**rf_params, random_state=42)
    elif model_type == "XGBoost":
        xgb_params = {k: v for k, v in params.items() if k in ['n_estimators', 'learning_rate', 'max_depth']}
        model = XGBClassifier(**xgb_params, eval_metric="logloss", random_state=42, verbosity=0)
    elif model_type == "Logistic Regression":
        lr_params = {k: v for k, v in params.items() if k in ['C', 'max_iter']}
        model = LogisticRegression(**lr_params, random_state=42)
    elif model_type == "SVM":
        svc_params = {k: v for k, v in params.items() if k in ['C', 'kernel']}
        model = SVC(**svc_params, probability=True, random_state=42)
    
    model.fit(X, y)
    return model

# ---------- 5. MAIN APP ----------
try:
    training_data = load_data()
    latest_race, latest_year = get_prediction_context()

    feature_cols = [
        "driver_sg_races", "driver_sg_wins", "driver_sg_avg_finish",
        "driver_sg_best_finish", "driver_sg_podiums", "driver_sg_recent_avg",
        "driver_sg_recent_podiums", "team_sg_races", "team_sg_wins",
        "team_sg_winrate", "team_sg_avg_finish", "team_sg_podiums",
    ]

    X = training_data[feature_cols]
    y = training_data["win"]
    X_latest = latest_race[feature_cols]

    # --- LAYOUT ---
    col_settings, col_results = st.columns([1, 2.5], gap="large")

    # --- LEFT PANEL: CONFIGURATION ---
    with col_settings:
        st.markdown(f"""
        <div>
            <h3 style="margin-top:0; margin-bottom: 1rem;">Configuration</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Appearance
        st.caption("APPEARANCE")
        st.checkbox("Dark Mode", key="dark_mode")
        
        # Divider
        st.markdown(f"<div style='margin: 4px 0; border-top: 1px solid {theme['border']};'></div>", unsafe_allow_html=True)

        # Model Selection
        st.caption("MODEL")
        model_type = st.selectbox(
            "Select Model", 
            ["XGBoost", "Random Forest", "Logistic Regression", "SVM"], 
            label_visibility="collapsed"
        )
        
        # Parameters
        st.markdown("<div style='margin: 8px 0;'></div>", unsafe_allow_html=True)
        st.caption("TUNING")
        
        params = {}
        if model_type == "XGBoost":
            lr = st.slider("Learning Rate", 0.05, 0.2, 0.1, 0.01)
            params = {"n_estimators": 100, "learning_rate": lr, "max_depth": 6}
        elif model_type == "Random Forest":
            trees = st.slider("Trees", 100, 300, 150, 50)
            params = {"n_estimators": trees, "max_depth": 10}
        elif model_type == "Logistic Regression":
            c_val = st.slider("Regularization (C)", 0.01, 10.0, 1.0, 0.1)
            params = {"C": c_val, "max_iter": 1000}
        else:
            c_val = st.slider("C Parameter", 0.1, 10.0, 1.0, 0.1)
            params = {"C": c_val, "kernel": "rbf", "probability": True}
        
        st.markdown("<div style='margin: 1.5rem 0'></div>", unsafe_allow_html=True)
        
        # Actions
        train_clicked = st.button("Generate Prediction", use_container_width=True)

        if st.session_state.get("trained", False):
            if st.button("Reset View", use_container_width=True):
                st.session_state["trained"] = False
                st.rerun()

    # --- RIGHT PANEL: DASHBOARD ---
    with col_results:
        
        st.title("Singapore GP Predictor")
        st.markdown(f"AI-Driven Race Analysis based on **{latest_year} Season Data**.")
        st.markdown(f"<div style='margin-bottom: 2rem; border-top: 1px solid {theme['border']};'></div>", unsafe_allow_html=True)
        
        if train_clicked:
            with st.spinner("Processing..."):
                model = train_model(model_type, X, y, **params)
                probs = model.predict_proba(X_latest)[:, 1]
                st.session_state.update({
                    "trained": True, 
                    "predictions": probs, 
                    "current_model": model_type
                })
                st.rerun()

        # VIEW A: Results
        if st.session_state.get("trained", False):
            st.subheader("Prediction Results")
            st.caption(f"Model: {st.session_state['current_model']} | Ranked by Probability")
            
            res_df = latest_race[["driver_name", "team_name"]].copy()
            res_df["win_prob"] = st.session_state["predictions"] * 100
            res_df = res_df.sort_values("win_prob", ascending=False).reset_index(drop=True)
            
            top_5 = res_df.head(5)
            max_val = top_5["win_prob"].max()

            for _, row in top_5.iterrows():
                pct = row['win_prob']
                width = (pct / max_val) * 100 if max_val > 0 else 0
                
                st.markdown(f"""
                <div class="driver-card">
                    <div style="display: flex; justify-content: space-between; align-items: flex-end;">
                        <div>
                            <div style="font-size: 1.1rem; font-weight: 700; color: {theme['text_primary']}">{row['driver_name']}</div>
                            <div style="font-size: 0.85rem; color: {theme['text_secondary']}; text-transform: uppercase;">{row['team_name']}</div>
                        </div>
                        <div style="font-size: 1.4rem; font-weight: 700; color: {theme['text_primary']}">{pct:.1f}%</div>
                    </div>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width: {width}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button("Download Report CSV", res_df.to_csv(index=False), "prediction_report.csv", "text/csv")

        # VIEW B: Dashboard
        else:
            st.subheader("Historical Overview")
            
            m1, m2, m3 = st.columns(3)
            with m1: st.metric("Total Races", str(training_data["year"].nunique()))
            with m2: st.metric("Drivers", str(training_data["driver_name"].nunique()))
            with m3: st.metric("Teams", str(training_data["team_name"].nunique()))
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Recent Winners")
            winners = training_data[training_data["win"] == 1].sort_values("year", ascending=False).head(4)
            
            for _, row in winners.iterrows():
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 1rem 1.5rem; background-color: {theme['bg_panel']}; border-radius: 8px; margin-bottom: 0.8rem; border: 1px solid {theme['border']};">
                   <div>
                       <span style="font-weight: 700; color: {theme['text_primary']}; font-size: 1.1rem; margin-right: 1.5rem;">{int(row['year'])}</span>
                       <span style="color: {theme['text_primary']}; font-weight: 600;">{row['driver_name']}</span>
                   </div>
                   <div style="color: {theme['text_secondary']}; font-size: 0.9rem; text-transform: uppercase;">{row['team_name']}</div>
                </div>
                """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"System Error: {str(e)}")
    st.info("Please verify the dataset exists in the 'data' directory.")