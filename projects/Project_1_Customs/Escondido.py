import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from Resultats import R0

# ── PLOTLY THEME ───────────────────────────────────────────────────────────────
PLOT_BG = "rgba(5,13,26,0)"
GRID_COL = "rgba(13,158,175,0.12)"
TEXT_COL = "#8ab4d0"
FONT_FAM = "JetBrains Mono, monospace"


class E0:
    def __init__(
        self,
        arrival_df,
        average_df,
        Tau: float,
        cN: int,
        cT: int,
    ):

        self.arrival_df = arrival_df
        self.average_df = average_df
        self.tau = Tau
        self.cN = cN
        self.cT = cT

        self.opt = R0(self.arrival_df, self.average_df, self.tau, self.cN, self.cT)
        self.N_opt = self.opt.N_opt
        self.N_naive = self.opt.N_naive
        self.N_long = self.opt.N_long
        self.Q_opt = self.opt.Q_opt
        self.T = len(self.Q_opt)

        self.best_T = self.opt.best_T
        self.best_N = self.opt.best_N

        # params :
        self.arrivals = self.opt.arrivals
        self.len = len(self.arrivals)
        self.labels = self.time_labels(self.len)

    def to_hist(self, Q):
        Q_int = np.floor(Q).astype(int)
        counts = np.bincount(Q_int)
        return counts

    def sum(self, L):
        dL = 0
        for i in range(len(L)):
            for j in range(len(L[0])):
                dL += L[i][j]
        return dL

    def get_results(self):
        st.title("Exploration des compromis")
        st.markdown(
            """
            <div class="section-label">
                Résultats de l'optimisation
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ===== KPI principaux =====
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(label="Temps optimal", value=f"{round(self.best_T)} min")

        with col2:
            st.metric(label="Nombre total de douaniers", value=self.sum(self.best_N))

        with col3:
            st.metric(
                label="Nombre total de douaniers actuel", value=self.sum(self.N_opt)
            )

        st.divider()

        # ===== Tableau des allocations =====
        st.markdown("### Répartition des effectifs")

        self.N1_best = [self.best_N[i][0] for i in range(len(self.best_N))]
        self.N2_best = [self.best_N[i][1] for i in range(len(self.best_N))]
        self.N3_best = [self.best_N[i][2] for i in range(len(self.best_N))]

        data = {
            "Créneau": [f"Créneau {i + 1}" for i in range(len(self.best_N))],
            "Tiers 1": self.N1_best,
            "Tiers 2": self.N2_best,
            "Tiers 3": self.N3_best,
        }

        df = pd.DataFrame(data)

        st.dataframe(df, use_container_width=True, hide_index=True)

        # ===== Histogramme / Bar chart =====
        st.markdown("### Visualisation des allocations")

        chart_df = pd.DataFrame(
            {
                "Créneau": [f"C{i + 1}" for i in range(len(self.best_N))],
                "T1_opt": self.N1_opt,
                "T2_opt": self.N2_opt,
                "T3_opt": self.N3_opt,
                "T1_best": self.N1_best,
                "T2_best": self.N2_best,
                "T3_best": self.N3_best,
            }
        )

        st.line_chart(chart_df.set_index("Créneau")[["T1_opt", "T1_best"]])
        st.line_chart(chart_df.set_index("Créneau")[["T2_opt", "T2_best"]])
        st.line_chart(chart_df.set_index("Créneau")[["T3_opt", "T3_best"]])

        # ===== Résumé texte =====
        st.success(
            f"""
            Optimisation terminée avec succès ! \n

            • Temps optimal trouvé : {round(self.best_T)} min \n
            • Effectif total optimisé : {self.sum(self.best_N)} douaniers\n
            • Effectif total actuel : {self.sum(self.N_opt)} douaniers\n
            
    
            
            
            """
        )

    def print_N(self):
        # ── Q function chart ──
        st.markdown(
            '<div class="section-label">Q(t) · Queue length function</div>',
            unsafe_allow_html=True,
        )
        fig_q = go.Figure()
        # Fill area
        fig_q.add_trace(
            go.Scatter(
                x=self.labels,
                y=self.N_long,
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(13,158,175,0.08)",
                line=dict(color="#0d9eaf", width=2.5),
                name="Queue length",
                hovertemplate="%{x} → <b>%{y:.1f}</b> pax queued<extra></extra>",
            )
        )
        # Arrivals overlay
        fig_q.add_trace(
            go.Scatter(
                x=self.labels,
                y=self.arrivals,
                mode="lines",
                line=dict(color="rgba(232,80,112,0.55)", width=1.5, dash="dot"),
                name="Arrivals",
                hovertemplate="%{x} → <b>%{y}</b> arrivals<extra></extra>",
            )
        )

        st.plotly_chart(fig_q, use_container_width=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown(
            f"""
                    <div style="text-align:center; font-family:'JetBrains Mono',monospace;
                                font-size:0.72rem; color:rgba(100,140,170,0.6); margin-top:1rem;">
                        OptiBorder · portfolio #01 · computed {len(self.arrivals)} slots ·
                        target wait {self.tau:.0f} min · R₀ algorithm
                    </div>
                    """,
            unsafe_allow_html=True,
        )
        # ===== Tableau des allocations =====
        st.markdown("### Répartition des effectifs Optimaux")
        self.N1_opt = [self.N_opt[i][0] for i in range(len(self.N_opt))]
        self.N2_opt = [self.N_opt[i][1] for i in range(len(self.N_opt))]
        self.N3_opt = [self.N_opt[i][2] for i in range(len(self.N_opt))]

        self.N1_naive = [self.N_naive[i][0] for i in range(len(self.N_opt))]
        self.N2_naive = [self.N_naive[i][1] for i in range(len(self.N_opt))]
        self.N3_naive = [self.N_naive[i][2] for i in range(len(self.N_opt))]

        data = {
            "Créneau": [f"Créneau {i + 1}" for i in range(len(self.N_opt))],
            "Tiers 1": self.N1_opt,
            "Tiers 2": self.N2_opt,
            "Tiers 3": self.N3_opt,
        }

        df = pd.DataFrame(data)

        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("### Répartition des effectifs Naifs")
        data_naive = {
            "Créneau": [f"Créneau {i + 1}" for i in range(len(self.N_opt))],
            "Tiers 1": self.N1_naive,
            "Tiers 2": self.N2_naive,
            "Tiers 3": self.N3_naive,
        }

        df_naive = pd.DataFrame(data_naive)

        st.dataframe(df_naive, use_container_width=True, hide_index=True)

    def print_Q(self):
        # ── Q function chart ──
        st.markdown(
            '<div class="section-label">Q(t) · Queue length function</div>',
            unsafe_allow_html=True,
        )

        fig_q = go.Figure()
        # Fill area
        fig_q.add_trace(
            go.Scatter(
                x=self.labels,
                y=self.Q_opt,
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(13,158,175,0.08)",
                line=dict(color="#0d9eaf", width=2.5),
                name="Queue length",
                hovertemplate="%{x} → <b>%{y:.1f}</b> pax queued<extra></extra>",
            )
        )
        # Arrivals overlay
        fig_q.add_trace(
            go.Scatter(
                x=self.labels,
                y=self.arrivals,
                mode="lines",
                line=dict(color="rgba(232,80,112,0.55)", width=1.5, dash="dot"),
                name="Arrivals",
                hovertemplate="%{x} → <b>%{y}</b> arrivals<extra></extra>",
            )
        )

        st.plotly_chart(fig_q, use_container_width=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="text-align:center; font-family:'JetBrains Mono',monospace;
                        font-size:0.72rem; color:rgba(100,140,170,0.6); margin-top:1rem;">
                OptiBorder · portfolio #01 · computed {len(self.arrivals)} slots ·
                target wait {self.tau:.0f} min · R₀ algorithm
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── N vector chart ──
        st.markdown(
            '<div class="section-label"> Distribution de Q </div>',
            unsafe_allow_html=True,
        )
        hist_Q = self.to_hist(self.Q_opt)
        x_vals = list(range(len(hist_Q)))  # cohérent avec histogramme

        fig_n = go.Figure()
        fig_n.add_trace(
            go.Bar(
                x=x_vals[1:],
                y=hist_Q[1:],
                name="Officers needed",
                marker=dict(
                    color=hist_Q,  # plus logique
                    colorscale=[[0, "#7b1a2a"], [0.5, "#9e3a5a"], [1, "#0d9eaf"]],
                    line=dict(width=0),
                ),
                hovertemplate="%{x} pax → <b>%{y:.0f}</b> amount by day <extra></extra>",
            )
        )
        fig_n.update_layout(
            **self.base_layout("Number of pax in the queue"), height=300
        )
        st.plotly_chart(fig_n, use_container_width=True)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── PLOTLY THEME ───────────────────────────────────────────────────────────────

    def base_layout(self, title=""):
        PLOT_BG = "rgba(5,13,26,0)"
        GRID_COL = "rgba(13,158,175,0.12)"
        TEXT_COL = "#8ab4d0"
        FONT_FAM = "JetBrains Mono, monospace"
        return dict(
            title=dict(
                text=title, font=dict(color="#c8d8f0", size=14, family=FONT_FAM)
            ),
            paper_bgcolor=PLOT_BG,
            plot_bgcolor=PLOT_BG,
            font=dict(color=TEXT_COL, family=FONT_FAM, size=11),
            xaxis=dict(
                gridcolor=GRID_COL, linecolor=GRID_COL, tickfont=dict(color=TEXT_COL)
            ),
            yaxis=dict(
                gridcolor=GRID_COL, linecolor=GRID_COL, tickfont=dict(color=TEXT_COL)
            ),
            margin=dict(l=40, r=20, t=50, b=40),
            hovermode="x unified",
        )

    def time_labels(self, n_slots):
        labels = []
        total_min = 0
        for i in range(n_slots):
            m = total_min + i * 15
            labels.append(f"{m // 60:02d}:{m % 60:02d}")
        return labels
