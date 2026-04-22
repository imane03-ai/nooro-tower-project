import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
import plotly.express as px
import os

# Configuration de la page
st.set_page_config(page_title="NOORo I Tower Tools", layout="wide")

st.title("🛠️ Outil d'Analyse Thermodynamique : NOORo I")

# --- CHARGEMENT DU MODÈLE IA ---
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model_path = os.path.join(os.getcwd(), 'modele_nooro_final.json')
    if os.path.exists(model_path):
        model.load_model(model_path)
        return model
    return None

model_ai = load_model()

# --- BARRE LATÉRALE ---
st.sidebar.header("📥 Données de la Centrale")
uploaded_file = st.sidebar.file_uploader("Charger le fichier Excel (10 min data)", type=['xlsx'])

# --- CALCULS ET ANALYSE ---
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df['time'] = pd.to_datetime(df['time'])
    
    # 1. Calculs au pas de 10 min
    df['Delta T'] = df['T_w_in'] - df['T_w_out_reel']
    
    # Estimation Twb (Stull) pour l'Approche
    T = df['T_db']
    Rh = df['HR']
    df['Twb'] = T * np.arctan(0.151977 * (Rh + 8.313659)**0.5) + np.arctan(T + Rh) - np.arctan(Rh - 1.676331) + 0.00391838 * (Rh)**1.5 * np.arctan(0.023101 * Rh) - 4.686035
    
    df['Approche'] = df['T_w_out_reel'] - df['Twb']
    df['Efficacite'] = (df['Delta T'] / (df['Delta T'] + df['Approche'])) * 100
    df['Evap_m3_h'] = 0.00153 * df['L'] * df['Delta T'] * 0.001

    # 2. Moyennes Journalières
    df_daily = df.set_index('time').resample('D').agg({
        'Delta T': 'mean',
        'Approche': 'mean',
        'Efficacite': 'mean',
        'Evap_m3_h': 'mean'
    }).reset_index()

    # --- AFFICHAGE ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Delta T Moyen", f"{round(df['Delta T'].mean(), 2)} °C")
    col2.metric("Approche Moyenne", f"{round(df['Approche'].mean(), 2)} °C")
    col3.metric("Efficacité Moyenne", f"{round(df['Efficacite'].mean(), 1)} %")
    col4.metric("Évaporation Totale", f"{round(df['Evap_m3_h'].sum()*(10/60), 1)} m³")

    # Graphique 1 : Delta T & Approche
    st.subheader("🌡️ Évolution des Températures (10 min)")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df['time'], y=df['Delta T'], name="Delta T", line=dict(color='firebrick')))
    fig1.add_trace(go.Scatter(x=df['time'], y=df['Approche'], name="Approche", line=dict(color='royalblue')))
    st.plotly_chart(fig1, use_container_width=True)

    # Graphique 2 : Évaporation Journalière
    st.subheader("📅 Consommation d'eau moyenne par jour")
    fig2 = px.bar(df_daily, x='time', y='Evap_m3_h', color='Evap_m3_h',
                 labels={'Evap_m3_h': 'Débit moyen (m³/h)', 'time': 'Date'})
    st.plotly_chart(fig2, use_container_width=True)

    # Bouton d'exportation
    st.sidebar.markdown("---")
    csv = df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button("📥 Télécharger les calculs (.CSV)", csv, "resultats_nooro.csv", "text/csv")

else:
    st.info("👋 Veuillez charger votre fichier Excel dans la barre latérale pour activer l'outil.")

# --- SECTION PRÉDICTION IA (Si modèle chargé) ---
if model_ai:
    st.sidebar.markdown("---")
    st.sidebar.header("🔮 Prédiction IA")
    # (Ajoutez ici vos curseurs de prédiction si vous voulez les garder)
