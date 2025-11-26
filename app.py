import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

st.set_page_config(page_title="Singapore GP Predictor", page_icon="🏁", 
                   layout="centered", initial_sidebar_state="expanded")

st.markdown("""<style>
    .main-title {font-size: 2.5rem; font-weight: 600; color: #1f1f1f; margin-bottom: 0.5rem;}
    .subtitle {font-size: 1.1rem; color: #666; margin-bottom: 2rem;}
    .stButton>button {width: 100%; border-radius: 6px; height: 3rem; font-weight: 500;}
</style>""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">🏁 Singapore GP Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict F1 Singapore GP winners with ML</p>', unsafe_allow_html=True)

@st.cache_data
def load_training_data():
    return pd.read_csv("data/training_dataset.csv")

@st.cache_data
def get_latest_drivers():
    df = pd.read_csv("data/training_dataset.csv")
    latest_year = df['year'].max()
    return df[df['year'] == latest_year].copy(), latest_year

@st.cache_resource
def train_model(model_type, params, X, y):
    models_map = {
        "Random Forest": RandomForestClassifier(**params, random_state=42),
        "Logistic Regression": LogisticRegression(**params, max_iter=1000, random_state=42),
        "XGBoost": XGBClassifier(**params, eval_metric="logloss", random_state=42),
        "SVM": SVC(**params, probability=True, random_state=42)
    }
    model = models_map[model_type]
    model.fit(X, y)
    return model

try:
    training_data = load_training_data()
    latest_race, latest_year = get_latest_drivers()
    
    feature_cols = ['driver_sg_races', 'driver_sg_wins', 'driver_sg_avg_finish',
                    'driver_sg_best_finish', 'driver_sg_podiums', 'driver_sg_recent_avg',
                    'driver_sg_recent_podiums', 'team_sg_races', 'team_sg_wins',
                    'team_sg_winrate', 'team_sg_avg_finish', 'team_sg_podiums']
    
    with st.sidebar:
        st.header("Model Settings")
        model_type = st.selectbox("Algorithm", ["XGBoost", "Random Forest", 
                                                  "Logistic Regression", "SVM", "Ensemble"])
        st.divider()
        
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
        st.caption(f"📊 Data: {training_data['year'].min()}-{training_data['year'].max()}, "
                  f"{training_data['year'].nunique()} races, {len(training_data)} samples")
        st.divider()
        train_button = st.button("Train Model", type="primary", use_container_width=True)
    
    X, y = training_data[feature_cols], training_data['win']
    X_latest = latest_race[feature_cols]
    
    if train_button:
        with st.spinner("Training..."):
            if model_type == "Ensemble":
                models = [LogisticRegression(max_iter=1000, random_state=42),
                         RandomForestClassifier(n_estimators=100, random_state=42),
                         XGBClassifier(eval_metric="logloss", random_state=42),
                         SVC(probability=True, random_state=42)]
                preds = [m.fit(X, y).predict_proba(X_latest)[:, 1] for m in models]
                latest_probs = np.mean(preds, axis=0)
            else:
                model = train_model(model_type, params, X, y)
                latest_probs = model.predict_proba(X_latest)[:, 1]
            
            st.session_state.update({'trained': True, 'predictions': latest_probs,
                                     'model_type': model_type, 'params': params})
            st.success("Trained!")
            st.rerun()
    
    if st.session_state.get('trained', False):
        results = latest_race[['driver_name', 'team_name']].copy()
        results['win_probability'] = st.session_state['predictions'] * 100
        results = results.sort_values('win_probability', ascending=False).reset_index(drop=True)
        
        winner = results.iloc[0]
        st.markdown(f"### 🏆 Predicted Winner - {latest_year} Singapore GP")
        col1, col2, col3 = st.columns([2, 2, 1])
        col1.metric("Driver", winner['driver_name'])
        col2.metric("Team", winner['team_name'])
        col3.metric("Probability", f"{winner['win_probability']:.1f}%")
        st.caption(f"Model: {st.session_state.get('model_type')}")
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
        st.download_button("📥 Download Predictions", results.to_csv(index=False),
                          f"singapore_{latest_year}.csv", "text/csv", use_container_width=True)
        
        st.markdown("### 📈 Model Validation")
        with st.expander("Past Performance", expanded=False):
            years_to_show = sorted(training_data['year'].unique())[-3:]
            model_type_used = st.session_state.get('model_type')
            params_used = st.session_state.get('params', {})
            
            for year in years_to_show:
                year_data = training_data[training_data['year'] == year].copy()
                X_year = year_data[feature_cols]
                train_mask = training_data['year'] < year
                X_train_temp = training_data[train_mask][feature_cols]
                y_train_temp = training_data[train_mask]['win']
                
                if len(X_train_temp) > 10:
                    if model_type_used == "Ensemble":
                        temp_models = [LogisticRegression(max_iter=1000, random_state=42),
                                      RandomForestClassifier(n_estimators=100, random_state=42),
                                      XGBClassifier(eval_metric="logloss", random_state=42),
                                      SVC(probability=True, random_state=42)]
                        temp_preds = [m.fit(X_train_temp, y_train_temp).predict_proba(X_year)[:, 1] 
                                     for m in temp_models]
                        year_probs = np.mean(temp_preds, axis=0)
                    else:
                        temp_model = train_model(model_type_used, params_used, X_train_temp, y_train_temp)
                        year_probs = temp_model.predict_proba(X_year)[:, 1]
                    
                    year_data['predicted_prob'] = year_probs
                    top_3 = year_data.sort_values('predicted_prob', ascending=False).head(3)
                    st.markdown(f"**{year}:**")
                    for i, (_, row) in enumerate(top_3.iterrows(), 1):
                        mark = " ✅ **WINNER**" if row['win'] == 1 else ""
                        st.write(f"{i}. {row['driver_name']} - {row['predicted_prob']*100:.1f}%{mark}")
    
    else:
        st.info(f"👈 Configure settings and click **Train Model** to predict {latest_year} winner")
        st.markdown("### 📊 Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Races", training_data['year'].nunique())
        col2.metric("Years", f"{training_data['year'].min()}-{training_data['year'].max()}")
        col3.metric("Drivers", training_data['driver_name'].nunique())
        
        st.markdown("### 🏁 Recent Winners")
        for _, row in training_data[training_data['win'] == 1].sort_values('year', ascending=False).head(5).iterrows():
            st.write(f"**{int(row['year'])}:** {row['driver_name']} ({row['team_name']})")

except FileNotFoundError:
    st.error("⚠️ Data not found. Run: load_data.py → merge_data.py → create_training_data.py")

st.divider()
st.caption("Data: Kaggle F1 Championship (1950-2020)")
