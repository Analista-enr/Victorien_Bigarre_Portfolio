"""
Objectif de cette section :
    - faire tourner les modèles
    - optimiser les modèles
    - évaluer les résultats
        - distribution du dataset :
            - nombre de scaffold différents
        - résultats :
            - comparer les modèles
            - comparer les failles des modèles
"""


def EDA_statistique(self, dataset) -> pd.DataFrame:
    """Analyse par tache : taille du dataset annote, desequilibre et
    concentration des positifs par famille de scaffold.

    Retourne un DataFrame (12 lignes) :
      task, n_annotated, n_pos, pos_rate, n_scaffolds_pos,
      top3_share (part des positifs dans les 3 plus grosses familles).
    """
    # -- extraction directe depuis le dataset (plus d'argument Y externe) --
    Y = self.stack_labels(dataset)  # (n_mol, 12), NaN conserves

    # -- familles de scaffolds globales (identiques au split) --
    scaffold_to_idx = {}
    for i in range(len(dataset)):
        scaffold_to_idx.setdefault(self.murcko_scaffold_smiles(dataset[i]), []).append(
            i
        )

    rows = []
    for t, name in enumerate(TASK_NAMES):
        y_t = Y[:, t]
        annotated = ~np.isnan(y_t)
        n_annotated = int(annotated.sum())
        pos_idx = np.where(y_t == 1)[0]  # indices des positifs

        if len(pos_idx) == 0:  # tache sans aucun positif
            rows.append(
                {
                    "task": name,
                    "n_annotated": n_annotated,
                    "n_pos": 0,
                    "pos_rate": 0.0,
                    "n_scaffolds_pos": 0,
                    "top3_share": np.nan,
                }
            )
            continue

        # familles des molécules positives (une famille peut apparaître plusieurs
        # fois si plusieurs positifs la partagent — c'est justement le signal)
        pos_family_sizes = [
            len(scaffold_to_idx[self.murcko_scaffold_smiles(dataset[i])])
            for i in pos_idx
        ]
        top3_share = float(np.sort(pos_family_sizes)[-3:].sum() / len(pos_idx))

        rows.append(
            {
                "task": name,
                "n_annotated": n_annotated,  # molécules testées sur cette tâche
                "n_annotated_share": 100 * round(n_annotated / 7823, 2),
                "n_pos": len(pos_idx),
                "n_scaffolds_pos": len(
                    set(self.murcko_scaffold_smiles(dataset[i]) for i in pos_idx)
                ),
                "pos_rate (%) ": round(100 * len(pos_idx) / n_annotated, 2),
                "top3_share (%)": round(top3_share, 2),
            }
        )
    df = pd.DataFrame(rows)
    df["AUC_SVM"] = [
        np.float64(0.7637564381759371),
        np.float64(0.8346816508577228),
        np.float64(0.8383128855542126),
        np.float64(0.7379983135662949),
        np.float64(0.7036555204015106),
        np.float64(0.7790513681066347),
        np.float64(0.7516067023086201),
        np.float64(0.7392137017117733),
        np.float64(0.7877337811030495),
        np.float64(0.719773581848089),
        np.float64(0.8325107776183542),
        np.float64(0.751375456204357),
    ]
    df["AUC_SVM_Tanimoto"] = [
        np.float64(0.76636379624712),
        np.float64(0.8508728081174223),
        np.float64(0.8511887392169402),
        np.float64(0.7576396341109417),
        np.float64(0.7127062707483469),
        np.float64(0.78081826858475),
        np.float64(0.773847006025392),
        np.float64(0.7706591668322907),
        np.float64(0.8018759451236958),
        np.float64(0.7385897030080996),
        np.float64(0.8483839917277287),
        np.float64(0.7801640207335038),
    ]
    df["AUC_LGBM"] = [
        np.float64(0.7148657893720418),
        np.float64(0.811033341218333),
        np.float64(0.8271887973309026),
        np.float64(0.7522284341744109),
        np.float64(0.6753752610872512),
        np.float64(0.7300185968333347),
        np.float64(0.7382615477061526),
        np.float64(0.7477153334130355),
        np.float64(0.789480199880695),
        np.float64(0.7326278490200222),
        np.float64(0.823201484145039),
        np.float64(0.7536326493010692),
    ]

    df = df.set_index("task")

    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return df


def extract(
    self,
    df: pd.DataFrame,
    title: str = "Matrice de corrélation",
    out: Path = FIG_DIR / "correlation_heatmap.png",
):
    """Heatmap de correlation (RdBu), divergente autour de 0.
    Rouge = correlation positive, bleu = negative (centree sur zero)."""
    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots(
        figsize=(2 + 0.6 * len(corr.columns), 1.5 + 0.5 * len(corr.columns))
    )
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)

    # etiquettes : noms de variables sur les axes
    labels = corr.columns
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)

    # annotations : valeur dans chaque cellule
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                f"{corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(corr.iloc[i, j]) > 0.5 else "black",
            )

    fig.colorbar(im, ax=ax, label="Corrélation de Pearson", shrink=0.8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Figure sauvegardée : {out}")
    return fig


def scaffold_distribution(
    self, dataset, out: Path = FIG_DIR / "scaffold_distribution.png"
):
    """Distribution des molecules par scaffold.

    Retourne un DataFrame descriptif (une ligne par famille) et produit
    une figure a double volet :
      - gauche : histogramme des tailles de familles (echelle log!)
      - droite  : courbe de Pareto (part cumulee des molecules selon
                  les k plus grosses familles) — LA lecture cle.
    """
    # -- 1. familles : scaffold -> liste des indices de ses molecules --
    scaffold_to_idx = {}
    for i in range(len(dataset)):
        scaffold_to_idx.setdefault(self.murcko_scaffold_smiles(dataset[i]), []).append(
            i
        )

    sizes = np.array([len(v) for v in scaffold_to_idx.values()])

    # -- 2. statistiques descriptives --
    stats = {
        "n_molecules": len(dataset),
        "n_scaffolds": len(scaffold_to_idx),
        "mean_family_size": sizes.mean(),
        "median_family_size": np.median(sizes),
        "max_family_size": sizes.max(),
        "n_singletons": int((sizes == 1).sum()),  # familles d'UNE molecule
        "top10_share": np.sort(sizes)[-10:].sum()
        / len(dataset),  # part des 10 plus grosses
    }

    # -- 3. figure --
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # volet 1 : histogramme des tailles (log-x : sinon la longue trainee ecrase tout)
    ax1.hist(np.log10(sizes), bins=40, color="#4C72B0", edgecolor="white")
    ax1.set_xlabel("log10(taille de la famille)")
    ax1.set_ylabel("Nombre de familles")
    ax1.set_title(f"{stats['n_scaffolds']} familles de scaffolds")
    ax1.axvline(
        np.log10(np.median(sizes)),
        color="red",
        linestyle="--",
        label=f"mediane = {stats['median_family_size']:.0f}",
    )
    ax1.legend()

    # volet 2 : Pareto — part cumulee des molecules couverte par les k plus grosses familles
    sizes_sorted = np.sort(sizes)[::-1]
    cumul = np.cumsum(sizes_sorted) / len(dataset)
    ax2.plot(np.arange(1, len(sizes_sorted) + 1), cumul, color="#C44E52")
    ax2.axhline(0.8, color="gray", linestyle=":", label="80% des molecules")
    ax2.set_xlabel("k plus grosses familles")
    ax2.set_ylabel("Part cumulee des molecules")
    ax2.set_title("Courbe de Pareto : concentration du dataset")
    ax2.legend()

    fig.suptitle("Distribution des molecules par scaffold (Tox21)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Figure sauvegardee : {out}")

    for k, v in stats.items():
        print(f"{k:>22} : {v}")
    return stats

    # ------------------------------------------------------------------------- synthese
    def summarize(self, results: dict, out_dir: Path = Path("results")) -> Path:
        """Sauvegarde UNE version synthetique du run :
        - {model}_{split}_summary.json : metriques globales + params + horodatage,
        - {model}_{split}_per_task.csv : AUC par tache (moyenne +/- std sur seeds).
        Un run complet = 2 fichiers, prefixes par le nom du modele et le split.
        Retourne le chemin du JSON (la reference du run)."""
        out_dir.mkdir(parents=True, exist_ok=True)
        model = results["model_name"]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # -- 1. perimetre : val seule, ou val + test --
        scopes = {"val": results["val"]}
        if "test" in results:
            scopes["test"] = results["test"]

        # -- 2. CSV par tache : une ligne par (task), colonnes val/test --
        df = pd.DataFrame({"task": TASK_NAMES})
        for scope, res in scopes.items():
            if "auc_matrix" in res:  # moyenne +/- std sur seeds
                df[f"{scope}_auc"] = np.nanmean(res["auc_matrix"], axis=0)
                df[f"{scope}_auc_std"] = np.nanstd(res["auc_matrix"], axis=0)
            else:  # val en pipeline light
                df[f"{scope}_auc"] = res.get("auc_per_task", np.nan)
                df[f"{scope}_auc_std"] = np.nan
        csv_path = out_dir / f"{model}_{ts}_per_task.csv"
        df.to_csv(csv_path, index=False)

        # -- 3. JSON : le resume machine-lisible du run --
        summary = {
            "model_name": model,
            "params": results["params"],
            "n_seeds": results["n_seeds"],
            "timestamp": ts,
            "per_task_csv": csv_path.name,
        }
        for scope, res in scopes.items():
            summary[scope] = {
                "auc_mean": float(res["auc_mean"]),
                "auc_mean_std": float(res["auc_mean_std"]),
            }

        json_path = out_dir / f"{model}_{ts}_summary.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        # -- 4. lecture console compacte --
        print(f"\n===== {model} ({results['n_seeds']} seeds) =====")
        for scope, res in scopes.items():
            print(
                f"{scope:>5} AUC : {res['auc_mean']:.4f} +/- {res['auc_mean_std']:.4f}"
            )
        print(f"Resume : {json_path}")

        # -- 5. figure (test en priorite, sinon val) --
        plot_scope = "test" if "test" in results else "val"
        if "auc_matrix" in results[plot_scope]:
            self.plot_auc_per_task(results, scope=plot_scope)

        return json_path

    # ------------------------------------------------------------------------- graphe

    def plot_auc_per_task(
        self, results: dict, scope: str = "test", out: Path | None = None
    ) -> "plt.Figure":
        """Barres horizontales : AUC par tache (moyenne +/- std sur les seeds).
        Prend le dict de resultats complet + le scope ('val' ou 'test')."""
        res = results[scope]
        if "auc_matrix" not in res:
            raise ValueError(
                f"Pas de auc_matrix pour scope={scope!r} "
                f"(il faut evaluate_test=True ou plusieurs seeds)."
            )

        model_name = results["model_name"]
        if out is None:
            out = FIG_DIR / f"{model_name}_auc_per_task_{scope}.png"

        aucs = np.nanmean(res["auc_matrix"], axis=0)
        stds = np.nanstd(res["auc_matrix"], axis=0)
        order = np.argsort(aucs)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(
            np.array(TASK_NAMES)[order],
            aucs[order],
            xerr=stds[order],
            color="#4C72B0",
            alpha=1,
        )
        ax.axvline(0.5, color="red", linestyle="--", linewidth=1, label="Hasard (0.5)")
        ax.axvline(
            np.nanmean(aucs),
            color="orange",
            linestyle="--",
            linewidth=1,
            label=f"Moyenne ({np.nanmean(aucs):.3f})",
        )
        ax.set_xlim(0.4, 1)
        ax.set_xlabel(f"AUC ROC ({scope}, scaffold split)")
        ax.set_title(
            f"{model_name} + ECFP4 - AUC par tache ({results['n_seeds']} seeds)"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Figure sauvegardee : {out}")
        return fig
