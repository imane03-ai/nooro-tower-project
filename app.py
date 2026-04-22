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
    # On force le chemin pour être sûr qu'il le trouve à la racine
    import os
    model_path = os.path.join(os.getcwd(), 'modele_nooro_final.json')
    
    if os.path.exists(model_path):
        model.load_model(model_path)
        return model
    else:
        return None

model_ai = load_model()
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
    
    # --- CALCULS THERMODYNAMIQUES ---
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df['time'] = pd.to_datetime(df['time'])
    
    # --- 1. CALCULS TOUTES LES 10 MIN (Pas de temps de l'Excel) ---
    # Delta T (anciennement Range)
    df['Delta T'] = df['T_w_in'] - df['T_w_out_reel']
    
    # Calcul de l'Approche
    # Twb estimé par la formule de Stull (précise pour les pressions standards)
    T = df['T_db']
    Rh = df['HR']
    df['Twb'] = T * np.arctan(0.151977 * (Rh + 8.313659)**0.5) + np.arctan(T + Rh) - np.arctan(Rh - 1.676331) + 0.00391838 * (Rh)**1.5 * np.arctan(0.023101 * Rh) - 4.686035
    
    df['Approche'] = df['T_w_out_reel'] - df['Twb']
    
    # Efficacité (%)
    df['Efficacite'] = (df['Delta T'] / (df['Delta T'] + df['Approche'])) * 100
    
    # Évaporation instantanée (m3/h)
    df['Evap_m3_h'] = 0.00153 * df['L'] * df['Delta T'] * 0.001

    # --- 2. CALCUL DE L'ÉVAPORATION MOYENNE JOURNALIÈRE ---
    # On groupe par jour et on calcule la moyenne du débit d'évaporation
    df_daily = df.set_index('time').resample('D')['Evap_m3_h'].mean().reset_index()
    df_daily.columns = ['Jour', 'Evap_Moyenne_m3_h']

    # --- 3. AFFICHAGE DES RÉSULTATS ---
    st.header("📈 Analyse Détaillée de la Tour")
    
    # Tableau des indicateurs 10 min (Aperçu)
    st.subheader("⏱️ Indicateurs au pas de 10 min")
    st.dataframe(df[['time', 'Delta T', 'Approche', 'Efficacite', 'Evap_m3_h']].tail(10))

    # Graphique de l'Évaporation Moyenne par Jour
    st.subheader("📅 Évaporation Moyenne Journalière (m³/h)")
    fig_daily = px.bar(df_daily, x='Jour', y='Evap_Moyenne_m3_h', 
                       title="Consommation moyenne d'eau par jour",
                       labels={'Evap_Moyenne_m3_h': 'Moyenne m³/h'})
    st.plotly_chart(fig_daily, use_container_width=True)

    # Graphique Delta T vs Approche
    st.subheader("🌡️ Delta T et Approche (10 min)")
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(x=df['time'], y=df['Delta T'], name="Delta T (Saut thermique)"))
    fig_temp.add_trace(go.Scatter(x=df['time'], y=df['Approche'], name="Approche (Écart limite)"))
    fig_temp.update_layout(yaxis_title="Température (°C)", template="plotly_white")
    st.plotly_chart(fig_temp, use_container_width=True)
