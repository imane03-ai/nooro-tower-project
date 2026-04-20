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
    # 1. Range (Saut thermique) : Tw_in - Tw_out
    df['Range'] = df['T_w_in'] - df['T_w_out_reel']
    
    # 2. Approach : Tw_out - Twb (Température bulbe humide)
    # Note: Twb est estimée ici ou calculée via CoolProp si disponible
    # Pour simplifier dans l'immédiat, nous utilisons une approximation de Twb
    df['Approach'] = df['T_w_out_reel'] - df['T_db'] * (df['HR']/100)**(1/7) # Approximation simple

    # 3. Efficacité (%) : Range / (Range + Approach)
    df['Efficacite'] = (df['Range'] / (df['Range'] + df['Approach'])) * 100
    
    # 4. Pertes par Évaporation (m3/h) : Approximativement 0.00153 * L * Range * 0.001
    # Formule standard : 0.00085 * 1.8 * Range * Débit_eau
    df['Evaporation_m3_h'] = 0.00153 * df['L'] * df['Range'] * 0.001 
    
    # --- AFFICHAGE DES INDICATEURS (KPIs) ---
    st.subheader("📊 Indicateurs de Performance Moyens")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Range Moyen", f"{round(df['Range'].mean(), 2)} °C")
    col2.metric("Approach Moyen", f"{round(df['Approach'].mean(), 2)} °C")
    col3.metric("Efficacité", f"{round(df['Efficacite'].mean(), 1)} %")
    col4.metric("Évaporation Totale", f"{round(df['Evaporation_m3_h'].sum() * (10/60), 2)} m³")

    # --- GRAPHIQUE DES PERTES ---
    st.subheader("💧 Analyse des Pertes d'Eau")
    fig_pertes = go.Figure()
    fig_pertes.add_trace(go.Scatter(x=df['time'], y=df['Evaporation_m3_h'], name="Pertes Evaporation", fill='tozeroy'))
    fig_pertes.update_layout(title="Débit d'évaporation au cours du temps (m³/h)", template="plotly_white")
    st.plotly_chart(fig_pertes, use_container_width=True)

else:
    st.info("👋 Veuillez charger votre fichier Excel dans la barre latérale pour démarrer l'analyse.")
