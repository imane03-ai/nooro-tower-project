import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
import plotly.express as px
import os

# --- CONFIGURATION DE L'INTERFACE ---
st.set_page_config(page_title="NOORo I Tower Expert", layout="wide")
st.title("🚀 Système Expert d'Audit Thermique - NOORo I")
st.markdown("---")

# --- CHARGEMENT DU MODÈLE IA ---
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    # Assurez-vous que le nom du fichier correspond à celui sur votre GitHub
    path = os.path.join(os.getcwd(), 'modele_nooro_final.json')
    if os.path.exists(path):
        model.load_model(path)
        return model
    return None

model_ai = load_model()

# --- PARAMÈTRES FIXES DE LA CENTRALE ---
L_FIXE = 23600  # Débit d'eau constant (m3/h)
NB_VENTILATEURS = 8
CP_EAU = 4.186  # kJ/kg.K

# --- IMPORTATION DES DONNÉES ---
st.sidebar.header("📥 Données d'Exploitation")
uploaded_file = st.sidebar.file_uploader("Charger le fichier Excel (Pas de 10 min)", type=['xlsx'])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')

    # --- 1. MOTEUR DE CALCUL THERMODYNAMIQUE ---
    # Delta T (Nommé selon votre demande)
    df['Delta T'] = df['T_w_in'] - df['T_w_out_reel']
    
    # Température du bulbe humide (Twb) - Formule de Stull
    T, Rh = df['T_db'], df['HR']
    df['Twb'] = T * np.arctan(0.151977 * (Rh + 8.313659)**0.5) + np.arctan(T + Rh) - np.arctan(Rh - 1.676331) + 0.00391838 * (Rh)**1.5 * np.arctan(0.023101 * Rh) - 4.686035
    
    # Approche et Efficacité
    df['Approche'] = df['T_w_out_reel'] - df['Twb']
    df['Efficacite'] = (df['Delta T'] / (df['Delta T'] + df['Approche'])) * 100
    
    # Chaleur Rejetée (MW)
    df['Chaleur_MW'] = (L_FIXE * CP_EAU * df['Delta T']) / 3600
    
    # Perte par Évaporation (m3/h)
    df['Evap_m3_h'] = 0.00153 * L_FIXE * df['Delta T'] * 0.001

    # --- 2. PRÉDICTION IA ---
    if model_ai:
        try:
            # Utilisation des valeurs brutes pour éviter les erreurs de noms de colonnes
            features = df[['T_w_in', 'T_db', 'HR', 'L', 'G']]
            df['T_w_out_predite'] = model_ai.predict(features.values)
        except:
            st.sidebar.error("⚠️ Erreur : Colonnes manquantes pour l'IA (T_w_in, T_db, HR, L, G)")

    # --- 3. SYNTHÈSE JOURNALIÈRE ---
    df_daily = df.set_index('time').resample('D').agg({
        'Delta T': 'mean',
        'Approche': 'mean',
        'Efficacite': 'mean',
        'Evap_m3_h': 'mean',
        'Chaleur_MW': 'mean'
    }).reset_index()

    # --- 4. AFFICHAGE DU DASHBOARD ---
    # KPIs Globaux
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Efficacité Moyenne", f"{round(df['Efficacite'].mean(), 1)} %")
    k2.metric("Chaleur Totale (Moy)", f"{round(df['Chaleur_MW'].mean(), 2)} MW")
    k3.metric("Delta T Moyen", f"{round(df['Delta T'].mean(), 2)} °C")
    k4.metric("Eau évaporée (Moy)", f"{round(df['Evap_m3_h'].mean(), 1)} m³/h")

    # --- 5. SECTION DIAGNOSTIC INTELLIGENT ---
    st.header("📝 Diagnostic et Commentaires d'Expert")
    
    for index, row in df_daily.iterrows():
        dt = row['Delta T']
        evap = row['Evap_m3_h']
        date_label = row['time'].strftime('%d/%m/%Y')
        
        with st.expander(f"Analyse du {date_label}"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.write(f"**Performance Thermique :**")
                # --- 5. SECTION DIAGNOSTIC INTELLIGENT ---
    st.header("📝 Diagnostic et Commentaires d'Expert")
    
    for i in range(len(df_daily)):
        # On extrait les valeurs scalaires proprement
        dt = float(df_daily.loc[i, 'Delta T'])
        evap = float(df_daily.loc[i, 'Evap_m3_h'])
        date_label = df_daily.loc[i, 'time'].strftime('%d/%m/%Y')
        
        with st.expander(f"Analyse du {date_label}"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.write(f"**Performance Thermique :**")
                # Maintenant dt est un float, la comparaison fonctionne :
                if 8.0 <= dt <= 10.0:
                    st.success(f"✅ Delta T optimal : {round(dt,2)}°C. Échange conforme au design.")
                elif dt < 8.0:
                    st.error(f"🚨 Delta T faible : {round(dt,2)}°C. Les 8 ventilateurs ne suffisent pas à évacuer la charge (Air trop chaud).")
                else:
                    st.warning(f"🌡️ Delta T élevé : {round(dt,2)}°C. Refroidissement intense dû à un air très sec.")
            
            with col_b:
                st.write(f"**Analyse de l'eau :**")
                seuil_critique = L_FIXE * 0.015
                if evap > seuil_critique:
                    st.error(f"💧 Perte par évaporation élevée : {round(evap,1)} m³/h.")
                else:
                    st.info(f"💧 Évaporation normale : {round(evap,1)} m³/h.")

    # --- 6. VISUALISATIONS ---
    st.header("📈 Graphiques d'Analyse")
    
    # Graphique Températures
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(x=df['time'], y=df['T_w_out_reel'], name="Sortie Réelle", line=dict(color='blue')))
    if 'T_w_out_predite' in df.columns:
        fig_temp.add_trace(go.Scatter(x=df['time'], y=df['T_w_out_predite'], name="Sortie Prédite (IA)", line=dict(dash='dash', color='orange')))
    fig_temp.update_layout(title="Comparaison Températures de Sortie", yaxis_title="°C")
    st.plotly_chart(fig_temp, use_container_width=True)

    # Graphique Évaporation Journalière
    fig_evap = px.bar(df_daily, x='time', y='Evap_m3_h', title="Moyenne d'évaporation par jour (m³/h)", color='Evap_m3_h', color_continuous_scale='Blues')
    st.plotly_chart(fig_evap, use_container_width=True)

    # --- 7. EXPORT ---
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Télécharger le rapport complet (.csv)", csv, "audit_nooro_final.csv", "text/csv")

else:
    st.info("👋 Bienvenue. Veuillez charger votre fichier Excel pour lancer l'analyse automatique de la tour.")
