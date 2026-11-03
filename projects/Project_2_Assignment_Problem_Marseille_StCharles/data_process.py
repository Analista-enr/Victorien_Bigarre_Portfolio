from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator

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
        color_palette: str = "tab20",
        show_legend: bool = False,
        train_text_size: int = 10,
        hour_text_size: int = 11,
        bar_height: float = 0.62,
        alpha: float = 0.95,
        show_hour_lines: bool = True,
        highlight_conflicts: bool = True,
        show_night: bool = True,
        show_times: bool = False,
        title: Optional[str] = None,
        show: bool = False,
    ):
        """
        Diagramme de Gantt des affectations trains -> quais.

        Args:
            solution: affectations (index du train -> quai). Si None, utilise
                les affectations du DataFrame.
            highlight_conflicts: bordure rouge + hachures sur les trains en conflit.
            show_night: grise les plages de nuit (22h-6h).
            show_times: affiche l'heure d'arrivée / départ sous chaque barre.
            show: appelle plt.show() si True.

        Returns:
            (fig, ax)
        """
        if solution is None:
            sol = self.df["Affectation"].to_numpy().astype(int)
        else:
            sol = self._as_solution(solution)

        # --- Détection des conflits ------------------------------------
        conflict_mask = np.zeros(self.nb_train, dtype=bool)
        if highlight_conflicts and (sol > 0).any():
            same = (
                (sol[:, None] == sol[None, :]) & (self.overlap > 0) & (sol[:, None] > 0)
            )
            conflict_mask = same.any(axis=1)

        horizon = max(24 * 60, int(np.ceil(self.end.max() / 120) * 120))

        # --- Palette ----------------------------------------------------
        cmap = plt.get_cmap(color_palette)
        if hasattr(cmap, "colors") and len(cmap.colors) <= 30:  # palette discrète
            colors = [cmap(i % cmap.N) for i in range(max(self.nb_train, 1))]
        else:  # palette continue
            colors = [cmap(c) for c in np.linspace(0.05, 0.95, max(self.nb_train, 1))]

        def text_color(rgba):
            r, g, b = to_rgb(rgba)
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            return "#1f2933" if lum > 0.6 else "white"

        # --- Couleurs de thème -----------------------------------------
        BG = "#ffffff"
        INK = "#1f2933"
        MUTED = "#6b7785"
        GRID = "#d9dee4"
        BAND = "#f5f7fa"
        NIGHT = "#e8ecf2"
        BAD = "#d64545"

        fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
        ax.set_facecolor(BG)

        # --- Fond : bandes de quais + nuit ------------------------------
        for q in range(1, self.nb_quais + 1):
            if q % 2 == 0:
                ax.axhspan(q - 0.5, q + 0.5, color=BAND, zorder=0, lw=0)

        if show_night:
            for d in range(int(np.ceil(horizon / 1440))):
                ax.axvspan(
                    d * 1440, d * 1440 + 360, color=NIGHT, alpha=0.6, zorder=0.5, lw=0
                )
                ax.axvspan(
                    d * 1440 + 1320,
                    min((d + 1) * 1440, horizon),
                    color=NIGHT,
                    alpha=0.6,
                    zorder=0.5,
                    lw=0,
                )

        # --- Grille -----------------------------------------------------
        if show_hour_lines:
            ax.xaxis.set_minor_locator(MultipleLocator(60))
            ax.grid(axis="x", which="major", color=GRID, linewidth=1.0, zorder=1)
            ax.grid(
                axis="x", which="minor", color=GRID, linewidth=0.5, alpha=0.5, zorder=1
            )
        ax.set_axisbelow(True)

        # --- Géométrie pour des coins arrondis homogènes ----------------
        n_rows = self.nb_quais + 0.4
        aspect = (n_rows / figsize[1]) / (horizon / figsize[0])
        min_per_inch = horizon / (figsize[0] * 0.85)

        # --- Barres -----------------------------------------------------
        for i, num in enumerate(self.df["Numero"]):
            quai = int(sol[i])
            start = float(self.start[i])
            duration = float(self.end[i] - self.start[i])
            if quai == 0 or duration <= 0:
                continue

            bad = bool(conflict_mask[i])
            radius = min(0.07 * min_per_inch, duration / 2.5)

            # ombre portée légère
            ax.add_patch(
                FancyBboxPatch(
                    (start + horizon * 0.0012, quai - bar_height / 2 - 0.025),
                    duration,
                    bar_height,
                    boxstyle=f"round,pad=0,rounding_size={radius}",
                    mutation_aspect=aspect,
                    facecolor="black",
                    edgecolor="none",
                    alpha=0.10,
                    zorder=2,
                )
            )
            # barre principale
            ax.add_patch(
                FancyBboxPatch(
                    (start, quai - bar_height / 2),
                    duration,
                    bar_height,
                    boxstyle=f"round,pad=0,rounding_size={radius}",
                    mutation_aspect=aspect,
                    facecolor=colors[i],
                    edgecolor=BAD if bad else "white",
                    linewidth=2.4 if bad else 1.2,
                    alpha=alpha,
                    hatch="////" if bad else None,
                    zorder=3,
                )
            )

            # numéro du train : on réduit la police si la barre est étroite
            fs = train_text_size
            needed_min = (
                len(str(num)) * 0.62 * fs / 72 * min_per_inch + 0.1 * min_per_inch
            )
            while needed_min > duration and fs > 6:
                fs -= 1
                needed_min = (
                    len(str(num)) * 0.62 * fs / 72 * min_per_inch + 0.1 * min_per_inch
                )

            ax.text(
                start + duration / 2,
                quai,
                str(num),
                ha="center",
                va="center",
                color=text_color(colors[i]),
                fontsize=fs,
                fontweight="bold",
                zorder=4,
                clip_on=True,
            )

            if show_times:

                def hhmm(m):
                    m = int(m)
                    return f"{(m // 60) % 24:02d}:{m % 60:02d}"

                ax.text(
                    start + duration / 2,
                    quai - bar_height / 2 - 0.04,
                    f"{hhmm(self.start[i])}–{hhmm(self.end[i])}",
                    ha="center",
                    va="top",
                    fontsize=7,
                    color=MUTED,
                    zorder=4,
                )

        # --- Axes -------------------------------------------------------
        ax.set_xlim(0, horizon)
        ax.set_ylim(self.nb_quais + 0.5, 0.5)  # quai 1 en haut
        ax.set_yticks(range(1, self.nb_quais + 1))
        ax.set_yticklabels(
            [f"voie {q}" for q in range(1, self.nb_quais + 1)],
            fontsize=11,
            fontweight="bold",
            color=INK,
        )
        ax.tick_params(axis="y", length=0, pad=10)

        def fmt(x, pos):
            h = int(x // 60)
            label = f"{h % 24:02d}h"
            return label + (f"\n+{h // 24}j" if h >= 24 and h % 24 == 0 else "")

        ax.set_xticks(range(0, horizon + 1, 120))
        ax.xaxis.set_major_formatter(FuncFormatter(fmt))
        ax.tick_params(
            axis="x",
            which="major",
            labelsize=hour_text_size,
            colors=MUTED,
            length=4,
            color=GRID,
        )
        ax.tick_params(axis="x", which="minor", length=2, color=GRID)
        ax.xaxis.set_ticks_position("bottom")

        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)

        # --- Titre + sous-titre ----------------------------------------
        n_placed = int((sol > 0).sum())
        n_conf = int(conflict_mask.sum())
        subtitle = (
            f"{n_placed}/{self.nb_train} trains affectés  ·  {self.nb_quais} voies"
        )
        subtitle += f"  ·  {n_conf} en conflit" if n_conf else "  ·  aucun conflit ✓"

        fig.text(
            0.015,
            0.975,
            title or "Affectation des trains aux voies",
            fontsize=18,
            fontweight="bold",
            color=INK,
            ha="left",
            va="top",
        )
        fig.text(
            0.015,
            0.93,
            subtitle,
            fontsize=11,
            color=BAD if n_conf else MUTED,
            ha="left",
            va="top",
        )

        # --- Légende ----------------------------------------------------
        handles = []
        if show_legend and self.nb_train <= 15:
            handles += [
                Patch(facecolor=colors[i], edgecolor="white", label=f"Train {num}")
                for i, num in enumerate(self.df["Numero"])
            ]
        if n_conf:
            handles.append(
                Patch(
                    facecolor="white",
                    edgecolor=BAD,
                    hatch="////",
                    linewidth=1.5,
                    label="Conflit",
                )
            )
        if show_night:
            handles.append(Patch(facecolor=NIGHT, label="Nuit (22h–6h)"))
        if handles:
            ax.legend(
                handles=handles,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.07),
                ncol=min(len(handles), 8),
                frameon=False,
                fontsize=9,
                labelcolor=INK,
            )

        fig.tight_layout(rect=(0, 0, 1, 0.91))

        if show:
            plt.show()
        return fig, ax
