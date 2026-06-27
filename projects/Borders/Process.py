import numpy as np
import pandas as pd


class P0:
    def __init__(
        self,
        arrival_df,
        average_df,
    ):
        self.df = pd.read_excel(arrival_df)
        self.df = pd.DataFrame(self.df)
        self.av = pd.read_excel(average_df)
        self.av = pd.DataFrame(self.av)
        self.dico = dict(zip(self.av["Departure"], self.av["Time"]))
        self.data = self.process(self.df, self.dico)
        self.week = self.data.columns[:-2]

    def process(self, df, dico, key_col=0, start_col=2):
        coef = df.iloc[:, key_col].map(dico).fillna(1)
        num_cols = df.columns[start_col:]
        df[num_cols] = df[num_cols].mul(coef, axis=0)
        df.drop("Departure", axis=1, inplace=True)
        df["Arrival time"] = pd.to_datetime(
            df["Arrival time"], format="%H:%M:%S", errors="coerce"
        )
        df["Hour"] = df["Arrival time"].dt.hour
        df["Minute"] = df["Arrival time"].dt.minute
        df.drop("Arrival time", axis=1, inplace=True)
        return df

    def llegada_fast(self):
        """
        Entry : df d'horaires d'arrivées
        return: df au pas 1min avec des diracs aux horaires d'arrivées
        """
        Q = np.zeros(self.T)
        hours = self.data["Hour"].values
        minutes = self.data["Minute"].values
        idx = (hours - self.start) * 60 + minutes
        for col in self.week:
            values = self.data[col].values
            np.add.at(Q, idx, values)
        return Q
