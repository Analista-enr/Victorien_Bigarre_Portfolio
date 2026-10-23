from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

REQUIRED_COLUMNS = ("Numero", "Arrival", "Departure")


class TrainScheduler:
    """
    Données et fonctions communes pour l'affectation des trains aux quais.

    Contient :
    - le chargement et la validation des données (trains, horaires, affectations fixes) ;
    - la matrice `overlap` (minutes de chevauchement entre deux trains) ;
    - le score `op_score` ;
    - une borne de concurrence et une solution initiale gloutonne ;
    - la visualisation `plot_gantt`.

    Convention : les quais sont numérotés de 1 à `nb_quais`, les trains sont
    indexés par leur position (0..nb_train-1) dans le DataFrame d'entrée
    (l'ordre d'entrée est conservé).

    Les horaires sont exprimés en minutes depuis minuit du premier jour
    (`self.start`, `self.end`). Un train qui passe minuit (départ < arrivée)
    voit son départ décalé d'un jour, et l'horizon peut donc dépasser 24 h.
    """

    def __init__(
        self,
        df_excel: pd.DataFrame,
        nb_quai: int,
        penalty: int = 10,
        arrival_margin: int = 5,
        departure_margin: int = 5,
        min_dwell: int = 20,
        overlap_offset: int = 1,
    ):
        """
        Args:
            df_excel: DataFrame avec colonnes ['Numero', 'Arrival', 'Departure']
                et optionnellement 'Affectation' (0 ou NaN = non fixé).
            nb_quai: nombre de quais disponibles (1 à nb_quai).
            penalty: coefficient de pénalité par minute de chevauchement.
            arrival_margin: marge (min) avant l'arrivée.
            departure_margin: marge (min) après le départ.
            min_dwell: durée minimale (min) d'occupation avant le départ.
            overlap_offset: ajouté à la durée de chevauchement (1 = comportement
                historique : deux trains qui se touchent comptent 1 minute).
        """
        if nb_quai < 1:
            raise ValueError("nb_quai doit être >= 1")
        self.nb_quais = int(nb_quai)
        self.penalty = penalty
        self.arrival_margin = arrival_margin
        self.departure_margin = departure_margin
        self.min_dwell = min_dwell
        self.overlap_offset = overlap_offset

        self.df = self.get_arrivals_df(df_excel)
        self.nb_train = len(self.df)

        self.start, self.end = self._compute_minutes()

        # Affectations fixes (index du train -> quai)
        self.fixed_assignments: Dict[int, int] = {
            int(i): int(q) for i, q in self.df["Affectation"].items() if q != 0
        }

        # Dictionnaire des horaires (numéro_train -> {...})
        self.dico = self._get_train_schedule()

        # Matrice des chevauchements (pré-calculée)
        self.overlap = self._compute_overlaps()

    # ------------------------------------------------------------------ #
    # Alias de compatibilité
    # ------------------------------------------------------------------ #
    @property
    def nb_quai(self) -> int:
        return self.nb_quais

    # ------------------------------------------------------------------ #
    # Chargement et validation
    # ------------------------------------------------------------------ #
    def get_arrivals_df(self, df_excel: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in REQUIRED_COLUMNS if c not in df_excel.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes : {missing}")

        df = df_excel.copy().reset_index(drop=True)

        for col in ("Arrival", "Departure"):
            df[col] = pd.to_datetime(df[col].astype(str), errors="coerce")

        bad = df[df[["Arrival", "Departure"]].isna().any(axis=1)]
        if not bad.empty:
            raise ValueError(
                "Horaires manquants ou illisibles pour les trains : "
                f"{bad['Numero'].tolist()}"
            )

        dup = df.loc[df["Numero"].duplicated(keep=False), "Numero"].unique()
        if len(dup) > 0:
            raise ValueError(f"Numéros de train dupliqués : {list(dup)}")

        # Train qui passe minuit : le départ est le lendemain
        crosses_midnight = df["Departure"] < df["Arrival"]
        df.loc[crosses_midnight, "Departure"] += pd.Timedelta(days=1)

        # Affectations fixes
        if "Affectation" not in df.columns:
            df["Affectation"] = 0
        df["Affectation"] = (
            pd.to_numeric(df["Affectation"], errors="coerce").fillna(0).astype(int)
        )
        out_of_range = df[(df["Affectation"] < 0) | (df["Affectation"] > self.nb_quais)]
        if not out_of_range.empty:
            raise ValueError(
                f"Affectation hors de [0, {self.nb_quais}] pour les trains : "
                f"{out_of_range['Numero'].tolist()}"
            )

        # Horaires réels avec marges
        df["Arrival_real"] = np.minimum(
            df["Arrival"] - pd.Timedelta(minutes=self.arrival_margin),
            df["Departure"] - pd.Timedelta(minutes=self.min_dwell),
        )
        df["Departure_real"] = df["Departure"] + pd.Timedelta(
            minutes=self.departure_margin
        )

        return df.drop(columns=["Arrival", "Departure"])

    def _compute_minutes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Minutes depuis minuit du premier jour (toujours >= 0)."""
        self.day0 = self.df["Arrival_real"].min().normalize()

        def to_min(s: pd.Series) -> np.ndarray:
            return (
                ((s - self.day0).dt.total_seconds() / 60).round().astype(int).to_numpy()
            )

        return to_min(self.df["Arrival_real"]), to_min(self.df["Departure_real"])

    def _get_train_schedule(self) -> Dict:
        """Dictionnaire numéro_train -> horaires et affectation."""
        dico = {}
        for i, num in enumerate(self.df["Numero"]):
            s, e = int(self.start[i]), int(self.end[i])
            dico[num] = {
                "arr": [(s // 60) % 24, s % 60],
                "dep": [(e // 60) % 24, e % 60],
                "start": s,
                "end": e,
                "emp": i,
                "affect": int(self.df.at[i, "Affectation"]),
            }
        return dico

    # ------------------------------------------------------------------ #
    # Chevauchements et score
    # ------------------------------------------------------------------ #
    def _compute_overlaps(self) -> np.ndarray:
        s, e = self.start, self.end
        ov = (
            np.minimum(e[:, None], e[None, :])
            - np.maximum(s[:, None], s[None, :])
            + self.overlap_offset
        )
        ov = np.maximum(ov, 0)
        np.fill_diagonal(ov, 0)
        return ov

    def _as_solution(self, N: Sequence[int]) -> np.ndarray:
        N = np.asarray(N).astype(int)
        if N.shape != (self.nb_train,):
            raise ValueError(
                f"La solution doit avoir {self.nb_train} éléments, reçu {N.shape}"
            )
        return N

    def op_score(self, N: Sequence[int]) -> int:
        """Pénalité totale : penalty * minutes de chevauchement sur un même quai."""
        N = self._as_solution(N)
        same = N[:, None] == N[None, :]
        return self.penalty * int((self.overlap * same).sum() // 2)

    def conflicts(self, N: Sequence[int]) -> List[Tuple[int, int, int]]:
        """Liste des (train_i, train_j, minutes) en conflit sur un même quai."""
        N = self._as_solution(N)
        same = (N[:, None] == N[None, :]) & (self.overlap > 0)
        ii, jj = np.nonzero(np.triu(same, 1))
        return [(int(i), int(j), int(self.overlap[i, j])) for i, j in zip(ii, jj)]

    def check_solution(self, N: Sequence[int]) -> None:
        """Lève une ValueError si la solution est invalide (plage, quais fixes)."""
        N = self._as_solution(N)
        if N.min() < 1 or N.max() > self.nb_quais:
            raise ValueError(f"Quais attendus dans [1, {self.nb_quais}]")
        for i, q in self.fixed_assignments.items():
            if N[i] != q:
                raise ValueError(f"Le train {i} devrait être sur le quai {q}")

    # ------------------------------------------------------------------ #
    # Outils d'analyse
    # ------------------------------------------------------------------ #
    def max_concurrency(self) -> int:
        """
        Nombre maximal de trains simultanément présents.
        Si <= nb_quais (et sans affectations fixes contraignantes), un score
        de 0 est atteignable.
        """
        thr = 1 - self.overlap_offset
        counts = [
            int(((self.start <= s) & (self.end - s >= thr)).sum()) for s in self.start
        ]
        return max(counts) if counts else 0

    def greedy_init(self) -> np.ndarray:
        """Solution gloutonne : trains triés par arrivée, quai le moins conflictuel."""
        sol = np.zeros(self.nb_train, dtype=int)
        for i, q in self.fixed_assignments.items():
            sol[i] = q
        for i in np.argsort(self.start, kind="stable"):
            if i in self.fixed_assignments:
                continue
            cost = [
                self.overlap[i][sol == q].sum() for q in range(1, self.nb_quais + 1)
            ]
            sol[i] = int(np.argmin(cost)) + 1
        return sol

    # ------------------------------------------------------------------ #
    # Visualisation
    # ------------------------------------------------------------------ #
    def plot_gantt(
        self,
        solution: Optional[Sequence[int]] = None,
        figsize: tuple = (16, 8),
        color_palette: str = "Set3",
        show_legend: bool = False,
        train_text_size: int = 10,
        hour_text_size: int = 11,
        bar_height: float = 0.6,
        alpha: float = 0.85,
        show_hour_lines: bool = True,
        highlight_conflicts: bool = True,
        title: Optional[str] = None,
        show: bool = False,
    ):
        """
        Diagramme de Gantt des affectations trains -> quais.

        Args:
            solution: affectations (index du train -> quai). Si None, utilise
                les affectations du DataFrame.
            highlight_conflicts: entoure en rouge les trains en conflit.
            show: appelle plt.show() si True.

        Returns:
            (fig, ax)
        """
        if solution is None:
            sol = self.df["Affectation"].to_numpy().astype(int)
        else:
            sol = self._as_solution(solution)

        conflict_mask = np.zeros(self.nb_train, dtype=bool)
        if highlight_conflicts and (sol > 0).any():
            same = (
                (sol[:, None] == sol[None, :]) & (self.overlap > 0) & (sol[:, None] > 0)
            )
            conflict_mask = same.any(axis=1)

        horizon = max(24 * 60, int(np.ceil(self.end.max() / 120) * 120))

        with plt.style.context("ggplot"):
            fig, ax = plt.subplots(figsize=figsize)

            cmap = plt.get_cmap(color_palette)
            colors = [cmap(c) for c in np.linspace(0, 1, max(self.nb_train, 1))]

            for i, num in enumerate(self.df["Numero"]):
                quai = int(sol[i])
                duration = int(self.end[i] - self.start[i])
                if quai == 0 or duration <= 0:
                    continue
                bad = bool(conflict_mask[i])
                ax.barh(
                    quai,
                    left=self.start[i],
                    width=duration,
                    height=bar_height,
                    color=colors[i],
                    edgecolor="#c0392b" if bad else "#2c3e50",
                    linewidth=2.0 if bad else 0.8,
                    alpha=alpha,
                )
                ax.text(
                    self.start[i] + duration / 2,
                    quai,
                    str(num),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=train_text_size,
                    fontweight="bold",
                    bbox=dict(
                        facecolor="black",
                        alpha=0.7,
                        edgecolor="none",
                        boxstyle="round,pad=0.3",
                        lw=0,
                    ),
                )

            ax.set_xlim(0, horizon)
            ax.set_ylim(0.3, self.nb_quais + 0.7)
            ax.set_yticks(range(1, self.nb_quais + 1))
            ax.set_yticklabels(
                [f"Quai {q}" for q in range(1, self.nb_quais + 1)],
                fontsize=11,
                fontweight="bold",
                color="#2c3e50",
            )
            ax.set_ylabel("Quai", fontsize=13, fontweight="bold", color="#2c3e50")

            def fmt(x, pos):
                h = int(x // 60)
                label = f"{h % 24}h"
                return label + (f" (+{h // 24}j)" if h >= 24 else "")

            ax.set_xticks(range(0, horizon + 1, 120))
            ax.xaxis.set_major_formatter(FuncFormatter(fmt))
            ax.tick_params(axis="x", labelsize=hour_text_size)
            ax.set_xlabel("Heure", fontsize=13, fontweight="bold", color="#2c3e50")

            if show_hour_lines:
                for x in range(0, horizon + 1, 120):
                    ax.axvline(
                        x, color="#34495e", linestyle="--", alpha=0.4, linewidth=1
                    )

            ax.set_title(
                title or "Diagramme de Gantt - Affectation des trains aux quais",
                fontsize=16,
                fontweight="bold",
                pad=20,
                color="#2c3e50",
            )

            if show_legend and self.nb_train <= 15:
                handles = [
                    Patch(
                        facecolor=colors[i],
                        edgecolor="#2c3e50",
                        label=f"Train {num}",
                        alpha=alpha,
                    )
                    for i, num in enumerate(self.df["Numero"])
                ]
                ax.legend(
                    handles=handles, loc="upper right", fontsize=9, title="Trains"
                )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout()

        if show:
            plt.show()
        return fig, ax
