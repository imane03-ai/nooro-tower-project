import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
import plotly.express as px
import os

# Configuration
st.set_page_config(page_title="NOORo I - Simulation Tool", layout="wide")
st.title("🚀 Outil de Prédiction et d'Audit Thermique NOORo I")

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
st.sidebar.header("📥 Importation des Données")
uploaded_file = st.sidebar.file_uploader("Charger le fichier Excel", type=['xlsx'])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df['time'] = pd.to_datetime(df['time'])
    
    if model_ai:
        try:
            features = df[['T_w_in', 'T_db', 'HR', 'L', 'G']]
            df['T_w_out_predite'] = model_ai.predict(features.values)
            st.sidebar.success("✅ Prédictions IA générées")
        except:
            st.sidebar.error("Vérifiez les colonnes de votre Excel")
    
    # Calculs
    df['Delta T'] = df['T_w_in'] - df['T_w_out_reel']
    T, Rh = df['T_db'], df['HR']
    df['Twb'] = T * np.arctan(0.151977 * (Rh + 8.313659)**0.5) + np.arctan(T + Rh) - np.arctan(Rh - 1.676331) + 0.00391838 * (Rh)**1.5 * np.arctan(0.023101 * Rh) - 4.686035
    df['Approche'] = df['T_w_out_reel'] - df['Twb']
    df['Efficacite'] = (df['Delta T'] / (df['Delta T'] + df['Approche'])) * 100
    df['Chaleur_Rejetee_MW'] = (df['L'] * 4.186 * df['Delta T']) / 3600
    df['Evap_m3_h'] = 0.00153 * df['L'] * df['Delta T'] * 0.001

    df_daily = df.set_index('time').resample('D').agg({'Evap_m3_h': 'sum', 'Chaleur_Rejetee_MW': 'mean', 'Efficacite': 'mean'}).reset_index()

    # Affichage KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Efficacité Moyenne", f"{round(df['Efficacite'].mean(), 1)} %")
    k2.metric("Chaleur Moyenne", f"{round(df['Chaleur_Rejetee_MW'].mean(), 2)} MW")
    k3.metric("Approche Moyenne", f"{round(df['Approche'].mean(), 2)} °C")
    k4.metric("Eau évaporée (Total)", f"{round(df['Evap_m3_h'].sum()*(10/60), 0)} m³")

    # Graphes
    st.subheader("🎯 Comparaison Réel vs Prédit")
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=df['time'], y=df['T_w_out_reel'], name="Réel"))
    if 'T_w_out_predite' in df.columns:
        fig_comp.add_trace(go.Scatter(x=df['time'], y=df['T_w_out_predite'], name="IA", line=dict(dash='dash')))
    st.plotly_chart(fig_comp, use_container_width=True)

    st.subheader("💧 Évaporation Journalière (m³)")
    st.plotly_chart(px.bar(df_daily, x='time', y='Evap_m3_h'), use_container_width=True)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Télécharger les résultats", csv, "audit_nooro.csv", "text/csv")

else:
    st.info("Veuillez charger le fichier Excel pour démarrer.")
