import pandas as pd


class Data:
    def __init__(self, camino1: str, camino2: str):
        # ✅ Définis d'abord les attributs
        self.camino1 = camino1
        self.camino2 = camino2
        # ✅ Puis appelle get_conso()
        self.conso = self.get_conso()

    def get_conso(self):
        try:
            df_2023 = pd.read_csv(self.camino1, sep=";", encoding="latin1")
            df_2024 = pd.read_csv(self.camino2, sep=";", encoding="latin1")

            df = pd.concat([df_2023, df_2024], ignore_index=True)
            df["cosh"] = cos

            df["DateHeure"] = df["Date"].astype(str) + " " + df["Heures"].astype(str)
            df["DateHeure"] = pd.to_datetime(df["DateHeure"], format="%d/%m/%Y %H:%M")

            df = df.drop(columns=["Date", "Heures"])

            df = df.set_index("DateHeure")

            df = df.ffill().interpolate(method="time").bfill()

            return df

        except FileNotFoundError as e:
            print(f"❌ Fichier introuvable : {e}")
            return None
        except Exception as e:
            print(f"❌ Erreur : {e}")
            return None
