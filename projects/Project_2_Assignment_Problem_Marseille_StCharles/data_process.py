import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt

class TrainScheduler:
    """
    Classe de base pour la gestion des données et des fonctions universelles.
    Contient :
    - Le chargement des données (trains, horaires, affectations fixes).
    - Le calcul de la matrice `overlap` (chevauchements temporels).
    - Le calcul du score `op_score`.
    - La visualisation `plot_gantt`.
    """

    def __init__(self, df_excel : pd.DataFrame, nb_quai: int, penalty: int = 10):
        """
        Args:
            df: DataFrame avec colonnes ['Numero', 'Arrival_real', 'Departure_real', 'Affectation'].
            nb_quai: Nombre de quais disponibles (1 à nb_quai).
            penalty: Coefficient de pénalité pour les chevauchements.
        """
        self.df = self.get_arrivals_df(df_excel)
        self.nb_quais = nb_quai
        self.nb_train = len(self.df)
        self.penalty = penalty
        self.T = 24 * 60  # 24h en minutes

        # Affectations fixes (train_index -> quai)
        self.fixed_assignments = {
            idx: int(row["Affectation"])
            for idx, row in self.df.iterrows()
            if row["Affectation"] != 0
        }

        # Dictionnaire des horaires (numéro_train -> {arr, dep, emp, affect})
        self.dico = self._get_train_schedule()

        # Matrice des chevauchements (pré-calculée)
        self.overlap = self._compute_overlaps()

    def get_arrivals_df(self, df_excel : pd.DataFrame):
        df = df_excel.copy()
        for col in ["Arrival", "Departure"]:
            df[col] = pd.to_datetime(df[col].astype(str))

        delta_20min = pd.to_timedelta("20min")
        delta_5min = pd.to_timedelta("5min")
        df["Arrival_real"] = np.minimum(
            df["Arrival"] - delta_5min, df["Departure"] - delta_20min
        )
        df["Departure_real"] = df["Departure"] + delta_5min

        df.drop(columns=['Arrival','Departure'], inplace=True)
        df["Affectation"].fillna(0)
        df.sort_values(by=["Numero"], ascending=False).reset_index(drop=True)
        return df

    def _get_train_schedule(self) -> Dict:
        """Retourne un dictionnaire avec les horaires et affectations de chaque train."""
        dico = {}
        for num in self.df["Numero"]:
            train_data = {}
            arrival = self.df.loc[self.df["Numero"] == num, "Arrival_real"].iloc[0]
            departure = self.df.loc[self.df["Numero"] == num, "Departure_real"].iloc[0]
            quai = self.df.loc[self.df["Numero"] == num, "Affectation"].iloc[0]
            train_data["arr"] = [arrival.hour, arrival.minute]
            train_data["dep"] = [departure.hour, departure.minute]
            train_data["emp"] = self.df[self.df["Numero"] == num].index[0]  # Index du train
            train_data["affect"] = quai
            dico[num] = train_data
        return dico

    def _compute_overlaps(self) -> List[List[int]]:
        """
        Calcule une matrice de chevauchements temporels entre tous les trains.
        overlap[i][k] = nombre de minutes où les trains i et k se chevauchent.
        """
        overlap = [[0] * self.nb_train for _ in range(self.nb_train)]
        for i in range(self.nb_train):
            num_i = self.df.iloc[i]["Numero"]
            deb_i = self.dico[num_i]["arr"][0] * 60 + self.dico[num_i]["arr"][1]
            fin_i = self.dico[num_i]["dep"][0] * 60 + self.dico[num_i]["dep"][1]

            for k in range(i + 1, self.nb_train):
                num_k = self.df.iloc[k]["Numero"]
                deb_k = self.dico[num_k]["arr"][0] * 60 + self.dico[num_k]["arr"][1]
                fin_k = self.dico[num_k]["dep"][0] * 60 + self.dico[num_k]["dep"][1]
                overlap_ik = max(0, min(fin_i, fin_k) - max(deb_i, deb_k) + 1)
                overlap[i][k] = overlap_ik
                overlap[k][i] = overlap_ik
        return overlap

    def op_score(self, N: List[int]) -> float:
        """
        Calcule le score pour une solution N.
        Score = penalty * Σ(overlap[i][k] * (N[i] == N[k])) pour i < k.
        """
        s = 0
        for i in range(self.nb_train):
            for k in range(i + 1, self.nb_train):
                s += self.overlap[i][k] * (N[i] == N[k])
        return s * self.penalty

    def plot_gantt(self, solution: List[int] = None, figsize: tuple = (16, 8),
                   color_palette: str = 'Set3', show_legend: bool = False,
                   train_text_size: int = 10, hour_text_size: int = 11,
                   bar_height: float = 0.6, alpha: float = 0.85,
                   show_hour_lines: bool = True, title: str = None):
        """
        Affiche un diagramme de Gantt **esthétique et professionnel** des affectations des trains aux quais.

        Args:
            solution: Liste d'affectations (train_index -> quai). Si None, utilise les affectations du DataFrame.
            figsize: Taille de la figure (largeur, hauteur). Défaut: (16, 8).
            color_palette: Palette de couleurs matplotlib ('Set3', 'tab20', 'viridis', 'pastel1').
            show_legend: Si True, affiche une légende avec les numéros de train.
            train_text_size: Taille du texte des numéros de train.
            hour_text_size: Taille du texte des repères horaires.
            bar_height: Hauteur des barres (0.5-0.8 recommandé).
            alpha: Transparence des barres (0.7-0.9 recommandé).
            show_hour_lines: Si True, affiche les lignes horaires.
            title: Titre personnalisé (défaut: titre générique).
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from matplotlib.ticker import FuncFormatter

        # --- Configuration initiale ---
        plt.figure(figsize=figsize)
        plt.style.use('ggplot')  # Style moderne et épuré

        # --- Palette de couleurs ---
        cmap = plt.get_cmap(color_palette)
        colors = [cmap(i) for i in np.linspace(0, 1, len(self.df["Numero"]))]
        color_map = {num: colors[i] for i, num in enumerate(self.df["Numero"])}

        # --- Tracer les barres pour chaque train ---
        for num in self.df["Numero"]:
            train_data = self.dico[num]
            start = train_data["arr"][0] * 60 + train_data["arr"][1]
            end = train_data["dep"][0] * 60 + train_data["dep"][1]
            duration = end - start

            # Déterminer le quai
            quai = solution[train_data["emp"]] if solution is not None else train_data["affect"]

            if quai != 0 and duration > 0:  # Ignorer les trains non affectés ou de durée nulle
                # Barre avec bordure noire et extrémités arrondies
                plt.barh(
                    quai,
                    left=start,
                    width=duration,
                    height=bar_height,
                    color=color_map[num],
                    edgecolor='#2c3e50',
                    linewidth=0.8,
                    alpha=alpha,
                    capstyle='round'  # Extrémités arrondies
                )

                # Texte du numéro de train (centré, avec fond semi-transparent)
                plt.text(
                    start + duration / 2,
                    quai,
                    str(num),
                    ha='center',
                    va='center',
                    color='white',
                    fontsize=train_text_size,
                    fontweight='bold',
                    bbox=dict(
                        facecolor='black',
                        alpha=0.7,
                        edgecolor='none',
                        boxstyle='round,pad=0.3',
                        lw=0
                    )
                )

        # --- Configuration des axes ---
        plt.xlim(0, 24 * 60)
        plt.ylim(0, self.nb_quais + 1.5)  # Espace pour les labels

        # Axe Y (quais) - Style amélioré
        plt.yticks(
            range(1, self.nb_quais + 1),
            [f'Quai {i}' for i in range(1, self.nb_quais + 1)],
            fontsize=11,
            fontweight='bold',
            color='#2c3e50'
        )
        plt.ylabel('Quai', fontsize=13, fontweight='bold', labelpad=15, color='#2c3e50')

        # Axe X (temps) - Format heures:minutes
        def minutes_to_hours(x, pos):
            hours = int(x // 60)
            minutes = int(x % 60)
            return f'{hours}h{minutes:02d}' if minutes else f'{hours}h'

        plt.gca().xaxis.set_major_formatter(FuncFormatter(minutes_to_hours))
        plt.xticks(
            range(0, 25 * 60, 2 * 60),  # Toutes les 2 heures
            fontsize=10,
            fontweight='bold',
            color='#2c3e50'
        )
        plt.xlabel('Heure', fontsize=13, fontweight='bold', labelpad=15, color='#2c3e50')

        # --- Titre personnalisable ---
        if title is None:
            title = '🚄 Diagramme de Gantt - Affectation Optimisée des Trains aux Quais'
        plt.title(
            title,
            fontsize=16,
            fontweight='bold',
            pad=20,
            color='#2c3e50'
        )

        # --- Lignes horaires (optionnelles) ---
        if show_hour_lines:
            for hour in range(0, 25, 2):
                plt.axvline(
                    x=hour * 60,
                    color='#34495e',
                    linestyle='--',
                    alpha=0.6,
                    linewidth=1
                )
                plt.text(
                    hour * 60,
                    self.nb_quais + 1.2,
                    f'{hour}h',
                    ha='center',
                    va='bottom',
                    fontsize=hour_text_size,
                    fontweight='bold',
                    color='#34495e',
                    bbox=dict(
                        facecolor='white',
                        alpha=0.8,
                        edgecolor='none',
                        boxstyle='round,pad=0.3'
                    )
                )

        # --- Grille discrète ---
        plt.grid(
            axis='x',
            linestyle=':',
            alpha=0.3,
            color='#ecf0f1',
            linewidth=0.8
        )

        # --- Légende (optionnelle) ---
        if show_legend:
            if len(self.df["Numero"]) <= 15:
                legend_elements = [
                    Patch(
                        facecolor=color_map[num],
                        edgecolor='#2c3e50',
                        label=f'Train {num}',
                        alpha=alpha,
                        linewidth=0.5
                    )
                    for num in self.df["Numero"]
                ]
                plt.legend(
                    handles=legend_elements,
                    loc='upper right',
                    bbox_to_anchor=(1.01, 1),
                    fontsize=9,
                    title='Légende des Trains',
                    title_fontsize=11,
                    framealpha=0.9,
                    edgecolor='#2c3e50'
                )
            else:
                plt.text(
                    0.99, 0.98,
                    f'Légende: {len(self.df["Numero"])} trains',
                    transform=plt.gca().transAxes,
                    ha='right',
                    va='top',
                    fontsize=9,
                    fontstyle='italic',
                    color='#7f8c8d',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
                )

        # --- Ajustements finaux pour un rendu pro ---
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.spines['left'].set_color('#2c3e50')
        ax.spines['bottom'].set_color('#2c3e50')
        ax.tick_params(axis='both', which='major', labelsize=10)

        plt.tight_layout()
        plt.show()