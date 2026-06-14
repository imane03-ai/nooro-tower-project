from supabase import create_client
import pandas as pd

url = "TON_URL_SUPABASE"
key = "TA_ANON_KEY"

supabase = create_client(url, key)

df = pd.read_excel("live_data.xlsx")

for _, row in df.iterrows():

    supabase.table("mesures").insert({
        "time": str(row["time"]),
        "T_w_in": float(row["T_w_in"]),
        "T_w_out_reel": float(row["T_w_out_reel"]),
        "T_db": float(row["T_db"]),
        "HR": float(row["HR"]),
        "L": float(row["L"]),
        "G": float(row["G"])
    }).execute()

print("Données envoyées")
