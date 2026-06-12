import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
import plotly.express as px
import os

# ==================================================
# CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="NOORo I Live Monitor",
    layout="wide"
)

st.title("🛰️ Système de Suivi en Temps Réel - NOORo I")

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

def get_data():

    fichiers = [
        "live_data.xlsx",
        "Live_data.xlsx",
        "live_data.xlsx.xlsx"
    ]

    for file in fichiers:
        if os.path.exists(file):
            return pd.read_excel(file)

    return None

df = get_data()

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

        except Exception as e:

            st.sidebar.warning(
                f"Erreur IA : {e}"
            )

    # ===============================================
    # KPI
    # ===============================================

    last_val = df.iloc[-1]

    niveau_col = None

    if "niveaux de bassin 1 dans CT %" in df.columns:
        niveau_col = "niveaux de bassin 1 dans CT %"

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
    # ===============================================
    # TEMPÉRATURES
    # ===============================================

    st.header(
        "📈 Suivi des Températures"
    )

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
                line=dict(
                    dash="dash"
                )
            )
        )

    st.plotly_chart(
    fig_temp,
    use_container_width=True,
    key="graph_temperature"
    )

    # ===============================================
    # ÉVAPORATION
    # ===============================================

    st.header(
        "💧 Flux d'Évaporation"
    )

    fig_evap = px.line(
        df,
        x="time",
        y="Evap_m3_h",
        title="Consommation d'eau"
    )

    st.plotly_chart(
    fig_evap,
    use_container_width=True,
    key="graph_evaporation"
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
