import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
import plotly.express as px
import os
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="NOORo I Live Monitor", layout="wide")

# --- PARAMÈTRES FIXES NOORo I ---
L_FIXE = 23600  # m3/h
NB_VENTILATEURS = 8

# --- FONCTION DE CONNEXION AUTOMATIQUE (SIMULATION API/SCADA) ---
def get_live_data():
    """
    Dans un système réel, cette fonction se connecte à votre base de données SQL 
    ou à votre système SCADA pour récupérer les données sans fichier Excel.
    """
    # Pour le moment, l'outil cherche un fichier 'live_data.xlsx' sur votre GitHub
    # qui se mettrait à jour automatiquement via un script externe.
    path = "live_data.xlsx.xlsx" 
    if os.path.exists(path):
        return pd.read_excel(path)
    else:
        return None

# --- CHARGEMENT IA ---
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    path = os.path.join(os.getcwd(), 'modele_nooro_final.json')
    if os.path.exists(path):
        model.load_model(path)
        return model
    return None

model_ai = load_model()

# --- INTERFACE ---
st.title("🛰️ Système de Monitoring en Temps Réel - NOORo I")
st.info(f"Connexion établie : Débit d'eau fixé à **{L_FIXE} m³/h** | **{NB_VENTILATEURS}** Ventilateurs en service.")

# Récupération automatique
df = get_live_data()

if df is not None:
    df['time'] = pd.to_datetime(df['time'])
    
    # --- CALCULS THERMODYNAMIQUES ---
    df['Delta T'] = df['T_w_in'] - df['T_w_out_reel']
    
    # Twb (Stull)
    T, Rh = df['T_db'], df['HR']
    df['Twb'] = T * np.arctan(0.151977 * (Rh + 8.313659)**0.5) + np.arctan(T + Rh) - np.arctan(Rh - 1.676331) + 0.00391838 * (Rh)**1.5 * np.arctan(0.023101 * Rh) - 4.686035
    
    df['Approche'] = df['T_w_out_reel'] - df['Twb']
    df['Efficacite'] = (df['Delta T'] / (df['Delta T'] + df['Approche'])) * 100
    
    # CALCUL ÉVAPORATION (Format 324 m3/h)
    # Formule : 0.00153 * L * DeltaT
    df['Evap_m3_h'] = 0.00153 * L_FIXE * df['Delta T']

    # Moyennes journalières
    df_daily = df.set_index('time').resample('D').agg({
        'Delta T': 'mean',
        'Evap_m3_h': 'mean',
        'Approche': 'mean'
    }).reset_index()

    # --- KPIs TEMPS RÉEL (Dernière valeur connue) ---
    last_val = df.iloc[-1]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Delta T Actuel", f"{round(last_val['Delta T'], 2)} °C")
    m2.metric("Évaporation Actuelle", f"{round(last_val['Evap_m3_h'], 1)} m³/h")
    m3.metric("Efficacité", f"{round(last_val['Efficacite'], 1)} %")
    m4.metric("Approche", f"{round(last_val['Approche'], 2)} °C")

    # --- DIAGNOSTIC AUTOMATIQUE ---
    st.header("📝 Diagnostic Automatique du Jour")
    current_dt = float(df_daily.iloc[-1]['Delta T'])
    current_evap = float(df_daily.iloc[-1]['Evap_m3_h'])

    col_diag1, col_diag2 = st.columns(2)
    with col_diag1:
        if 8 <= current_dt <= 10:
            st.success(f"✅ Delta T stable ({round(current_dt,1)}°C).")
        else:
            st.error(f"🚨 Anomalie Delta T ({round(current_dt,1)}°C) ! Vérifiez la charge thermique.")

    with col_diag2:
        if current_evap > 350: # Seuil d'alerte pour NOORo I
            st.warning(f"💧 Évaporation élevée : {round(current_evap,1)} m³/h. Attention au niveau d'appoint.")
        else:
            st.info(f"💧 Consommation d'eau nominale : {round(current_evap,1)} m³/h.")

    # --- GRAPHIQUE ---
    st.plotly_chart(px.line(df, x='time', y='Evap_m3_h', title="Flux d'évaporation (m³/h)"), use_container_width=True)

else:
    # Si aucun fichier 'live_data.xlsx' n'est trouvé, on laisse l'option manuelle
    st.warning("⚠️ Aucun flux de données en direct détecté. Veuillez charger un fichier pour tester.")
    manual_file = st.file_uploader("Charger manuellement")
    if manual_file:
        # Enregistrer le fichier pour simuler le "Live"
        with open("live_data.xlsx.xlsx", "wb") as f:
            f.write(manual_file.getbuffer())
        st.rerun()
