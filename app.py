import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="NOORo I - Cooling Tower Monitor", layout="wide")

st.title("🛡️ Dashboard de Performance : Tour de Refroidissement NOORo I")
st.markdown("""
Cette application permet le suivi thermodynamique en temps réel et la prédiction de performance 
de la tour de refroidissement, incluant l'impact des stratégies d'économie d'eau.
""")

# --- CHARGEMENT DU MODÈLE IA ---
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    # Assurez-vous que le fichier .json est dans le même dossier sur GitHub
    model.load_model('modele_nooro_final.json')
    return model

try:
    model_ai = load_model()
    model_loaded = True
except:
    model_loaded = False
    st.warning("⚠️ Modèle IA non trouvé. Veuillez uploader 'modele_nooro_final.json' sur GitHub.")

# --- BARRE LATÉRALE : INTERFACE D'UPLOAD ET PRÉDICTION ---
st.sidebar.header("📥 Données de la Centrale")
uploaded_file = st.sidebar.file_uploader("Charger le fichier Excel (10 min data)", type=['xlsx'])

st.sidebar.markdown("---")
st.sidebar.header("🔮 Prédiction pour Demain")
input_tw_in = st.sidebar.slider("Température entrée eau (T_w_in)", 30.0, 55.0, 45.0)
input_tdb = st.sidebar.slider("Température ambiante (T_db)", 10.0, 50.0, 38.0)
input_hr = st.sidebar.slider("Humidité Relative (HR %)", 5, 100, 15)
input_l = st.sidebar.number_input("Débit d'eau (L)", value=1200)
input_g = st.sidebar.number_input("Débit d'air (G)", value=800)

if st.sidebar.button("Prédire la performance"):
    if model_loaded:
        features = pd.DataFrame([[input_tw_in, input_l, input_g, input_hr, input_tdb]], 
                                columns=['T_w_in', 'L', 'G', 'HR', 'T_db'])
        prediction = model_ai.predict(features)[0]
        st.sidebar.success(f"Température de sortie prédite : {round(prediction, 2)} °C")
    else:
        st.sidebar.error("Erreur : Modèle non chargé.")

# --- CORPS PRINCIPAL : ANALYSE DES DONNÉES ---
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df['time'] = pd.to_datetime(df['time'])
    
    # Calculs techniques
    df['Economie_Eau_m3'] = (df['L'] * 0.0005) * (10/60) # Gain Drift Eliminator
    df['Delta_T'] = df['T_w_in'] - df['T_w_out_reel']
    
    # Affichage des KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Delta T Moyen", f"{round(df['Delta_T'].mean(), 2)} °C")
    col2.metric("Économie d'Eau Totale", f"{round(df['Economie_Eau_m3'].sum(), 2)} m³")
    col3.metric("Nombre de relevés", len(df))

    # Graphiques
    st.subheader("📈 Visualisation des Performances Historiques")
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("Suivi des Températures", "Économie d'eau cumulée (m³)"))

    fig.add_trace(go.Scatter(x=df['time'], y=df['T_w_in'], name="T_w_in", line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['T_w_out_reel'], name="T_w_out_réel", line=dict(color='blue')), row=1, col=1)
    
    df['Economie_Cumulee'] = df['Economie_Eau_m3'].cumsum()
    fig.add_trace(go.Scatter(x=df['time'], y=df['Economie_Cumulee'], name="Gain Drift Eliminator", fill='tozeroy'), row=2, col=1)

    fig.update_layout(height=700, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👋 Veuillez charger votre fichier Excel dans la barre latérale pour démarrer l'analyse.")
