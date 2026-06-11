import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
import plotly.express as px
import os

# --- CONFIGURATION ---
st.set_page_config(page_title="NOORo I Live Monitor", layout="wide")
st.title("🛰️ Système de Monitoring en Temps Réel - NOORo I")

# --- PARAMÈTRES FIXES NOORo I ---
L_FIXE = 23600  # m3/h
NB_VENTILATEURS = 8

# --- CHARGEMENT DU MODÈLE IA ---
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    path = os.path.join(os.getcwd(), 'modele_nooro_final.json')
    if os.path.exists(path):
        model.load_model(path)
        return model
    return None

model_ai = load_model()

# --- RÉCUPÉRATION DES DONNÉES (GitHub ou Manuel) ---
def get_data():
    for name in ["live_data.xlsx.xlsx", "Live_data.xlsx"]:
        if os.path.exists(name):
            return pd.read_excel(name)
    return None

df = get_data()

if df is not None:
    df['time'] = pd.to_datetime(df['time'])
    
    # --- CALCULS THERMODYNAMIQUES ---
    df['Delta T'] = df['T_w_in'] - df['T_w_out_reel']
    
    # Bulbe humide (Twb)
    T, Rh = df['T_db'], df['HR']
    df['Twb'] = T * np.arctan(0.151977 * (Rh + 8.313659)**0.5) + np.arctan(T + Rh) - np.arctan(Rh - 1.676331) + 0.00391838 * (Rh)**1.5 * np.arctan(0.023101 * Rh) - 4.686035
    
    df['Approche'] = df['T_w_out_reel'] - df['Twb']
    df['Efficacite'] = (df['Delta T'] / (df['Delta T'] + df['Approche'])) * 100
    df['Evap_m3_h'] = 0.00153 * L_FIXE * df['Delta T']

    # --- PRÉDICTION IA ---
    if model_ai:
        try:
            # On s'assure d'avoir les bonnes colonnes pour l'IA
            features = df[['T_w_in', 'T_db', 'HR', 'L', 'G']]
            df['T_w_out_predite'] = model_ai.predict(features.values)
        except:
            st.sidebar.warning("Colonnes L ou G manquantes pour l'IA")

    # --- MOYENNES JOURNALIÈRES ---
    df_daily = df.set_index('time').resample('D').agg({
        'Delta T': 'mean',
        'Evap_m3_h': 'mean',
        'Approche': 'mean',
        'Efficacite': 'mean'
    }).reset_index()

    # --- AFFICHAGE KPIs ---
    last_val = df.iloc[-1]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Delta T Actuel", f"{round(last_val['Delta T'], 2)} °C")
    m2.metric("Évaporation Actuelle", f"{round(last_val['Evap_m3_h'], 1)} m³/h")
    m3.metric("Efficacité", f"{round(last_val['Efficacite'], 1)} %")
    m4.metric("Approche", f"{round(last_val['Approche'], 2)} °C")

    # --- DIAGNOSTIC PAR JOUR (AVEC DATE) ---
    st.header("📝 Diagnostic et Commentaires d'Expert")
    for i in range(len(df_daily)):
        row = df_daily.iloc[i]
        date_str = row['time'].strftime('%d/%m/%Y')
        dt = float(row['Delta T'])
        evap = float(row['Evap_m3_h'])
        
        with st.expander(f"📅 Rapport du {date_str}"):
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**Delta T moyen :** {round(dt, 2)} °C")
                st.write(f"**Évaporation moyenne :** {round(evap, 1)} m³/h")
            with c2:
                if 8 <= dt <= 10:
                    st.success("✅ Delta T optimal.")
                elif dt < 8:
                    st.error("🚨 Delta T trop faible (Air trop chaud).")
                else:
                    st.warning("🌡️ Delta T élevé (Air très sec).")
                
                if evap > 350:
                    st.warning("💧 Consommation d'eau élevée.")

    # --- GRAPHIQUE DES TEMPÉRATURES (IA + RÉEL) ---
    st.header("📈 Suivi des Températures")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['time'], y=df['T_w_out_reel'], name="Température Réelle", line=dict(color='blue')))
    
    if 'T_w_out_predite' in df.columns:
        fig.add_trace(go.Scatter(x=df['time'], y=df['T_w_out_predite'], name="Prédiction IA", line=dict(color='orange', dash='dash')))
    
    fig.update_layout(title="Température de Sortie : Réel vs IA", yaxis_title="°C")
    st.plotly_chart(fig, use_container_width=True)

    # --- GRAPHIQUE ÉVAPORATION ---
    st.header("💧 Flux d'Évaporation")
    fig_evap = px.line(df, x='time', y='Evap_m3_h', title="Consommation d'eau en continu (m³/h)")
    st.plotly_chart(fig_evap, use_container_width=True)

else:
    st.warning("⚠️ Aucun fichier 'live_data.xlsx.xlsx' détecté sur GitHub.")
    uploaded = st.file_uploader("Charger manuellement pour tester")
    if uploaded:
        df_test = pd.read_excel(uploaded)
        st.success("Fichier chargé avec succès. Nommez-le 'live_data.xlsx.xlsx' sur GitHub pour l'automatisme.")
