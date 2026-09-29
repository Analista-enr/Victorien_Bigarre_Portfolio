import pandas as pd
import xlrd

class Datos:
    def __init__(self, camino_1: str, camino_2: str):
        self.df_demand = self.get_demand(camino_1)
        self.df_price = self.get_price(camino_2)

    def get_demand(self, path: str):
        df = pd.read_csv(path, encoding='latin1', sep=r"\t")
        return df

    def get_price(self, path: str):
        df = pd.read_excel(path, engine="openpyxl")
        return df
