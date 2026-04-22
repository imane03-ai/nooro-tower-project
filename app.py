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
uploaded_file = st.sidebar.file_uploader("Charger le fichier Excel de la centrale", type=['xlsx'])

if uploaded_file is not None:
    # Lecture des données
    df = pd.read_excel(uploaded_file)
    df['time'] = pd.to_datetime(df['time'])
    
# --- 1. PRÉDICTION IA POUR CHAQUE LIGNE ---
    if model_ai:
      try:
       # On sélectionne les colonnes nécessaires
       features = df[['T_w_in', 'T_db', 'HR', 'L', 'G']]
            
       # ASTUCE : On convertit en valeurs brutes (numpy) pour éviter l'erreur de noms de colonnes
         df['T_w_out_predite'] = model_ai.predict(features.values)
            
            st.sidebar.success("✅ Prédictions IA générées")
        except Exception as e:
            st.sidebar.error(f"Erreur colonnes : Vérifiez que votre Excel contient bien T_w_in, T_db, HR, L, G")
    
    # --- 2. CALCULS THERMODYNAMIQUES (10 MIN) ---
    # Delta T (Saut thermique)
    df['Delta T'] = df['T_w_in'] - df['T_w_out_reel']
    
    # Température humide (Twb) - Formule de Stull
    T, Rh = df['T_db'], df['HR']
    df['Twb'] = T * np.arctan(0.151977 * (Rh + 8.313659)**0.5) + np.arctan(T + Rh) - np.arctan(Rh - 1.676331) + 0.00391838 * (Rh)**1.5 * np.arctan(0.023101 * Rh) - 4.686035
    
    # Approche et Efficacité
    df['Approche'] = df['T_w_out_reel'] - df['Twb']
    df['Efficacite'] = (df['Delta T'] / (df['Delta T'] + df['Approche'])) * 100
    
    # Chaleur Rejetée Q (MW) = m * Cp * DeltaT / 3600 (pour passer de kJ/h à MW)
    # Cp eau = 4.186 kJ/kg.K, Masse volumique env 1000 kg/m3
    df['Chaleur_Rejetee_MW'] = (df['L'] * 4.186 * df['Delta T']) / 3600
    
    # Évaporation (m3/h)
    df['Evap_m3_h'] = 0.00153 * df['L'] * df['Delta T'] * 0.001

    # --- 3. ANALYSE JOURNALIÈRE ---
    df_daily = df.set_index('time').resample('D').agg({
        'Evap_m3_h': 'sum',
        'Chaleur_Rejetee_MW': 'mean',
        'Efficacite': 'mean'
    }).reset_index()

    # --- 4. AFFICHAGE DES RÉSULTATS ---
    # KPIs de synthèse
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Efficacité Moyenne", f"{round(df['Efficacite'].mean(), 1)} %")
    kpi2.metric("Chaleur Moyenne", f"{round(df['Chaleur_Rejetee_MW'].mean(), 2)} MW")
    kpi3.metric("Approche Moyenne", f"{round(df['Approche'].mean(), 2)} °C")
    kpi4.metric("Eau évaporée (Total)", f"{round(df['Evap_m3_h'].sum()*(10/60), 0)} m³")

    # Graphique de comparaison Réel vs Prédit
    st.subheader("🎯 Précision du Modèle : Température de Sortie Réelle vs Prédite")
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=df['time'], y=df['T_w_out_reel'], name="Réel", line=dict(color='blue')))
    if 'T_w_out_predite' in df.columns:
        fig_comp.add_trace(go.Scatter(x=df['time'], y=df['T_w_out_predite'], name="Prédit (IA)", line=dict(color='orange', dash='dash')))
    st.plotly_chart(fig_comp, use_container_width=True)

    # Graphique Évaporation par jour
    st.subheader("💧 Évaporation Totale Cumulée par Jour (m³)")
    fig_evap = px.bar(df_daily, x='time', y='Evap_m3_h', color_continuous_scale='Blues')
    st.plotly_chart(fig_evap, use_container_width=True)

    # Tableau de données avec toutes les colonnes calculées
    st.subheader("📋 Tableau des résultats calculés (Aperçu)")
    st.write(df[['time', 'T_w_out_reel', 'T_w_out_predite', 'Delta T', 'Approche', 'Efficacite', 'Chaleur_Rejetee_MW']].tail(10))

    # Bouton de téléchargement
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Télécharger le fichier complet avec calculs et prédictions", csv, "audit_nooro_complet.csv", "text/csv")

else:
    st.info("Veuillez charger le fichier Excel dans la barre latérale pour démarrer l'audit.")
