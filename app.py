import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
import plotly.express as px
import os
from streamlit_autorefresh import st_autorefresh
from supabase import create_client
from supabase import create_client
import streamlit as st
import pandas as pd

url = st.secrets["SUPABASE_URL"]
key = st.secrets["SUPABASE_KEY"]

supabase = create_client(url, key)

try:
    response = (
        supabase
        .table("mesures")
        .select("*")
        .limit(5)
        .execute()
    )

    st.success("Connexion Supabase OK")
    st.write(response.data)

except Exception as e:
    st.error(f"Erreur : {e}")

# ==================================================
# CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="NOORo I Live Monitor",
    layout="wide"
)

st.title("🛰️ Système de Suivi en Temps Réel - NOORo I")

st_autorefresh(
    interval=30000,
    key="auto_refresh"
)
# ==================================================
# PARAMÈTRES FIXES
# ==================================================

L_FIXE = 23600
NB_VENTILATEURS = 8

# ==================================================
# CHARGEMENT MODÈLE IA
# ==================================================

@st.cache_resource
def load_model():

    model = xgb.XGBRegressor()

    path = "modele_nooro_final.json"

    if os.path.exists(path):
        model.load_model(path)
        return model

    return None

model_ai = load_model()

# ==================================================
# CHARGEMENT DONNÉES
# ==================================================

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

st.write("URL utilisée :", SUPABASE_URL)
st.write("Début clé :", SUPABASE_KEY[:20])

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

try:

    response = (
        supabase
        .table("mesures")
        .select("*")
        .order("time")
        .execute()
    )

    df = pd.DataFrame(response.data)

except Exception as e:

    st.error(f"Erreur Supabase : {e}")

    st.stop()

df = df.sort_values("time")

last_val = df.iloc[-1]

st.write("Dernière date :", last_val["time"])

st.write("Dernier niveau bassin :", last_val.get("niveau_bassin"))

st.write("Nombre de lignes :", len(df))

st.write("Dernière mesure :")

st.dataframe(df.tail())
# ==================================================
# SI DONNÉES DISPONIBLES
# ==================================================

if df is not None:

    df.columns = df.columns.str.strip()

    # ===============================================
    # Vérification colonnes
    # ===============================================

    required_columns = [
        "time",
        "T_w_in",
        "T_w_out_reel",
        "T_db",
        "HR",
        "L",
        "G"
    ]

    missing = [c for c in required_columns if c not in df.columns]

    if len(missing) > 0:
        st.error(f"Colonnes manquantes : {missing}")
        st.stop()

    # ===============================================
    # Date
    # ===============================================

    df["time"] = pd.to_datetime(df["time"])

    # ===============================================
    # CALCULS
    # ===============================================
    df = df.dropna(
    subset=[
        "T_w_in",
        "T_w_out_reel",
        "T_db",
        "HR"
    ]
    )
    df["Delta T"] = df["T_w_in"] - df["T_w_out_reel"]

    T = df["T_db"]
    RH = df["HR"]

    df["Twb"] = (
        T * np.arctan(0.151977 * np.sqrt(RH + 8.313659))
        + np.arctan(T + RH)
        - np.arctan(RH - 1.676331)
        + 0.00391838 * RH**1.5 * np.arctan(0.023101 * RH)
        - 4.686035
    )

    df["Approche"] = df["T_w_out_reel"] - df["Twb"]

    df["Efficacite"] = np.where(
        (df["Delta T"] + df["Approche"]) != 0,
        (df["Delta T"] /
         (df["Delta T"] + df["Approche"])) * 100,
        np.nan
    )

    df["Evap_m3_h"] = (
        0.00153 *
        L_FIXE *
        df["Delta T"]
    )

    # ===============================================
    # IA
    # ===============================================

    if model_ai is not None:

        try:

            features = df[
                [
                    "T_w_in",
                    "T_db",
                    "HR",
                    "L",
                    "G"
                ]
            ]

            df["T_w_out_predite"] = model_ai.predict(
                features.values
            )
            if "T_w_out_predite" in df.columns:

                 erreur = abs(
                      last_val["T_w_out_reel"]
                      -
                      last_val["T_w_out_predite"]
            )

            st.metric(
               "Erreur IA",
                    f"{erreur:.2f} °C"
            )

        except Exception as e:

            st.sidebar.warning(
                f"Erreur IA : {e}"
            )

    # ===============================================
    # KPI
    # ===============================================

    last_val = df.iloc[-1]
    
    st.sidebar.success(
    f"Dernière mise à jour : "
    f"{pd.Timestamp.now().strftime('%d/%m/%Y %H:%M:%S')}"
    )

    niveau_col = None

    if "niveau_bassin" in df.columns:
        niveau_col = "niveau_bassin"

    a1, a2, a3, a4 = st.columns(4)

    a1.metric(
        "T° Entrée",
        f"{last_val['T_w_in']:.2f} °C"
    )

    a2.metric(
        "T° Sortie",
         f"{last_val['T_w_out_reel']:.2f} °C"
    )

    a3.metric(
        "Humidité",
         f"{last_val['HR']:.1f} %"
    )

    a4.metric(
        "T° Air",
         f"{last_val['T_db']:.2f} °C"
    )
    
    m1, m2, m3, m4, m5 = st.columns(5)
    
    m1.metric(
         "Delta T Actuel",
          f"{last_val['Delta T']:.2f} °C"
    )
    
    m2.metric(
        "Évaporation",
        f"{last_val['Evap_m3_h']:.1f} m³/h"
    )

    m3.metric(
        "Efficacité",
        f"{last_val['Efficacite']:.1f} %"
    )

    m4.metric(
        "Approche",
        f"{last_val['Approche']:.2f} °C"
    )

    if niveau_col:
        m5.metric(
            "Niveau Bassin",
        f"{last_val[niveau_col]:.1f} %"
      
    )

    # ==================================
    # ALERTES AUTOMATIQUES
    # ==================================

    alertes = []

    if last_val["Delta T"] < 8:
        alertes.append("🚨 Delta T faible")

    if last_val["Approche"] > 6:
        alertes.append("⚠️ Approche élevée")

    if last_val["Efficacite"] < 60:
        alertes.append("⚠️ Efficacité faible")

    if niveau_col:

       if last_val[niveau_col] < 20:
           alertes.append(
            "🚨 Niveau du bassin critique"
       )

    st.header("🚨 Alertes")

    if len(alertes) > 0:

        for a in alertes:
            st.error(a)

    else:

         st.success(
              "✅ Aucun problème détecté"
    )
    # ===============================================
    # BASSIN CT
    # ===============================================

    if niveau_col:

        st.header("🏞️ Niveau du Bassin CT")

        niveau_actuel = float(
            last_val[niveau_col]
        )

        c1, c2 = st.columns([1, 2])

        with c1:

            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=niveau_actuel,
                    number={"suffix": "%"},
                    title={"text": "Niveau Bassin"},
                    gauge={
                        "axis": {
                            "range": [0, 100]
                        },
                        "bar": {
                            "color": "blue"
                        },
                        "steps": [
                            {
                                "range": [0, 30],
                                "color": "red"
                            },
                            {
                                "range": [30, 70],
                                "color": "orange"
                            },
                            {
                                "range": [70, 100],
                                "color": "green"
                            }
                        ]
                    }
                )
            )

            st.plotly_chart(
                fig_gauge,
                use_container_width=True,
                key="jauge_bassin"
            )
        with c2:

            fig_level = px.line(
                df,
                x="time",
                y=niveau_col,
                title="Évolution du niveau du bassin"
            )

            st.plotly_chart(
                fig_level,
                use_container_width=True,
                key="courbe_bassin"
            )

        if niveau_actuel < 20:
            st.error(
                "🚨 RISQUE D'ARRÊT : niveau bassin très faible"
            )

        elif niveau_actuel < 60:
            st.warning(
                "⚠️ Niveau moyen du bassin"
            )

        else:
            st.success(
                "✅ Niveau normal du bassin"
            )

    # ===============================================
    # MOYENNES JOURNALIÈRES
    # ===============================================

    df_daily = (
        df
        .set_index("time")
        .resample("D")
        .agg({
            "Delta T": "mean",
            "Evap_m3_h": "mean",
            "Approche": "mean",
            "Efficacite": "mean"
        })
        .reset_index()
    )
    # ===============================================
    # DIAGNOSTICS
    # ===============================================
    # ==================================
    # DIAGNOSTIC GLOBAL
    # ==================================

    st.header("🤖 Diagnostic IA")

    diagnostic = ""

    if last_val["Delta T"] < 8:

        diagnostic += (
           "Refroidissement insuffisant. "
        )

    if last_val["Approche"] > 6:

        diagnostic += (
           "Possible encrassement du packing. "
        )

    if last_val["Efficacite"] < 60:

         diagnostic += (
            "Performance thermique faible. "
         )

    if niveau_col:

       if last_val[niveau_col] < 30:

           diagnostic += (
              "Vérifier l'appoint d'eau. "
           )

    if diagnostic == "":

        diagnostic = (
           "Fonctionnement normal."
        )

    st.info(diagnostic)

    st.header("📝 Diagnostic et Commentaires d'Expert")

    for _, row in df_daily.iterrows():

        if pd.isna(row["time"]):
            continue

        date_str = row["time"].strftime("%d/%m/%Y")

        dt = row["Delta T"]
        evap = row["Evap_m3_h"]

        with st.expander(f"📅 Rapport du {date_str}"):

            if pd.notna(dt):
                st.write(f"**Delta T moyen :** {dt:.2f} °C")
            else:
                st.write("**Delta T moyen :** Non disponible")

            if pd.notna(evap):
                st.write(f"**Évaporation moyenne :** {evap:.1f} m³/h")
            else:
                st.write("**Évaporation moyenne :** Non disponible")

            if pd.notna(dt):

                if 8 <= dt <= 10:
                    st.success("✅ Delta T optimal")

                elif dt < 8:
                    st.error("🚨 Delta T trop faible")

                else:
                    st.warning("🌡️ Delta T élevé")

    # ===============================================
    # TEMPÉRATURES
    # ===============================================

    st.header("📈 Suivi des Températures")

    fig_temp = go.Figure()

    fig_temp.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["T_w_out_reel"],
            name="Réelle"
        )
    )

    if "T_w_out_predite" in df.columns:

        fig_temp.add_trace(
            go.Scatter(
                x=df["time"],
                y=df["T_w_out_predite"],
                name="IA",
                line=dict(dash="dash")
            )
        )

    st.plotly_chart(
        fig_temp,
        use_container_width=True
    )

    # ===============================================
    # ÉVAPORATION
    # ===============================================

    st.header("💧 Flux d'Évaporation")

    fig_evap = px.line(
        df,
        x="time",
        y="Evap_m3_h",
        title="Consommation d'eau"
    )

    st.plotly_chart(
        fig_evap,
        use_container_width=True
    )

# ==================================================
# PAS DE DONNÉES
# ==================================================

else:

    st.warning(
        "⚠️ Aucun fichier Excel trouvé"
    )

    uploaded = st.file_uploader(
        "Charger un fichier Excel",
        type=["xlsx"]
    )

    if uploaded:

        df_test = pd.read_excel(
            uploaded
        )

        st.success(
            "Fichier chargé avec succès"
        )

        st.dataframe(
            df_test.head()
        )
