import numpy as np
import pandas as pd
import datetime as datetime

nzejndkez
class TchouTchouData:
    def __init__(self, camino: str):
        self.arrivals = self.get_arrivals(camino)

    def get_arrivals(self, camino: str):
        df = pd.read_excel(camino)

        # 1. Convertis les colonnes "Heure:Minutes" en objets `timedelta`
        for col in ["Arrival", "Departure"]:
            # Si les données sont au format "HH:MM" (ex: "10:30")
            df[col] = pd.to_timedelta(df[col].astype(str))

        # 2. Définis les deltas (20 min et 5 min)
        delta_20min = pd.to_timedelta("20min")
        delta_5min = pd.to_timedelta("5min")

        # 3. Applique les opérations (soustraction/addition de durées)
        df["Arrival_real"] = np.minimum(
            df["Arrival"] - delta_20min, df["Departure"] - delta_20min
        )
        df["Departure_real"] = df["Departure"] + delta_5min

        # 4. (Optionnel) Convertis les résultats en format "HH:MM" pour l'affichage
        for col in ["Arrival_real", "Departure_real"]:
            df[col] = df[col].dt.components.apply(
                lambda x: f"{x.hours:02d}:{x.minutes:02d}", axis=1
            )

        return df
