import pandas as pd
import numpy as np
import time
from datetime import datetime

FICHIER = "live_data.xlsx"

while True:

    try:

        df = pd.read_excel(FICHIER)

        nouvelle_ligne = {
            "time": datetime.now(),

            "T_w_in": round(np.random.uniform(40, 45), 2),

            "T_w_out_reel": round(np.random.uniform(30, 35), 2),

            "T_db": round(np.random.uniform(25, 42), 2),

            "HR": round(np.random.uniform(20, 80), 2),

            "L": 23600,

            "G": round(np.random.uniform(150000, 220000), 0),

            "niveaux de bassin 1 dans CT %":
            round(np.random.uniform(40, 90), 1)
        }

        df = pd.concat(
            [df, pd.DataFrame([nouvelle_ligne])],
            ignore_index=True
        )

        df.to_excel(FICHIER, index=False)

        print(
            f"Nouvelle mesure ajoutée : "
            f"{datetime.now()}"
        )

    except Exception as e:

        print(e)

    # 10 minutes
    time.sleep(600)
