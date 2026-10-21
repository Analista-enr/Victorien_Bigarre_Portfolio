"""Evaluation des modeles entraines par models.py (AUC ROC masquee par tache).

Entree : une liste de Run (voir models.py). Un Run doit exposer :
    run.seed, run.split (nom du split), run.val, run.test (indices),
    run.model.name, run.model.params, run.model.predict(data, idx) -> (len(idx), 12)
evaluate.py ne connait donc RIEN des masques NaN du train ni de la Gram : il ne depend
que de predict(). Tout modele qui respecte cette interface (SVM, LGBM, plus tard le GAT)
est evalue de la meme facon.

Sortie : un tableau LONG (un DataFrame), une ligne par (modele, split, seed, subset, tache) :
    model, split, seed, subset ('val' | 'test'), task, auc
Tout le reste (moyennes, ecarts-types, comparaison, figures) s'en deduit.

Protocole : par defaut on n'evalue QUE la validation. Le test se demande explicitement
(subsets=("val", "test")) : l'optimisation d'hyperparametres ne doit jamais le toucher.

Usage typique :
    runs = Gaia(data).train("LGBM", seed_list=range(10))
    df = evaluate_runs(data, runs, subsets=("val", "test"))
    print(summarize(df))
    save_results(df, runs)
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sauvegarde sans affichage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from dataset import TASK_NAMES

RESULTS_DIR = Path("results")
FIG_DIR = RESULTS_DIR / "figures"
SUBSETS = ("val", "test")


# ================================================================== metrique
def masked_auc(Y, scores) -> np.ndarray:
    """AUC par tache. Y : (n, 12) labels avec NaN = non teste ; scores : (n, 12).
    Retourne (12,). NaN si la tache n'a pas deux classes parmi les molecules annotees,
    ou si le modele n'a pas de score pour cette tache (colonne NaN)."""
    assert Y.shape == scores.shape, f"Y {Y.shape} != scores {scores.shape}"
    aucs = np.full(Y.shape[1], np.nan)
    for t in range(Y.shape[1]):
        ok = ~np.isnan(Y[:, t]) & ~np.isnan(scores[:, t])  # annotee ET scoree
        y = Y[ok, t].astype(int)
        if len(np.unique(y)) < 2:  # AUC indefinie sans deux classes
            continue
        aucs[t] = roc_auc_score(y, scores[ok, t])
    return aucs


# ================================================================ evaluation
def evaluate_runs(data, runs, subsets=("val",)) -> pd.DataFrame:
    """Evalue chaque Run sur les subsets demandes. Peut melanger plusieurs modeles :
    evaluate_runs(data, runs_lgbm + runs_svm). Retourne le tableau long."""
    for s in subsets:
        if s not in SUBSETS:
            raise ValueError(f"subset inconnu : {s!r} (choix : {SUBSETS})")

    rows = []
    for run in runs:
        for subset in subsets:
            idx = getattr(run, subset)
            scores = run.model.predict(data, idx)
            aucs = masked_auc(data.labels[idx], scores)
            rows.extend(
                {
                    "model": run.model.name,
                    "split": run.split,
                    "seed": run.seed,
                    "subset": subset,
                    "task": task,
                    "auc": auc,
                }
                for task, auc in zip(TASK_NAMES, aucs)
            )
    df = pd.DataFrame(rows)

    n_nan = int(df["auc"].isna().sum())
    if n_nan:  # tache non apprise ou sans deux classes : visible, jamais silencieux
        bad = df[df["auc"].isna()].groupby(["model", "subset", "task"]).size()
        print(f"Attention : {n_nan} AUC indefinies (ignorees dans les moyennes) :\n{bad}")
    return df


# ================================================================ agregation
def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par (model, split, subset) :
    auc_mean = moyenne sur les seeds de l'AUC moyenne sur les taches,
    auc_std  = ecart-type entre seeds (ddof=1 ; NaN s'il n'y a qu'une seed),
    n_seeds  = nombre de seeds.
    Les AUC indefinies (NaN) sont ignorees dans la moyenne sur les taches."""
    per_seed = df.groupby(["model", "split", "subset", "seed"])["auc"].mean().rename("auc_seed")
    out = per_seed.reset_index().groupby(["model", "split", "subset"])["auc_seed"]
    return out.agg(auc_mean="mean", auc_std="std", n_seeds="count").reset_index()


def per_task_table(df: pd.DataFrame, subset: str = "test") -> pd.DataFrame:
    """AUC moyenne (sur les seeds) par tache : lignes = taches (ordre TASK_NAMES),
    colonnes = (model, split). C'est le tableau de comparaison des modeles."""
    d = df[df["subset"] == subset]
    table = d.groupby(["task", "model", "split"])["auc"].mean().unstack(["model", "split"])
    return table.reindex(TASK_NAMES)


# ============================================================ sauvegarde / lecture
def _json_default(o):
    """Types numpy -> types Python pour json.dump."""
    return o.item() if hasattr(o, "item") else str(o)


def save_results(df: pd.DataFrame, runs, out_dir=RESULTS_DIR) -> None:
    """Un couple de fichiers par (modele, split), nommes sans horodatage (un nouveau run
    ECRASE le precedent : un seul resultat de reference par modele et par split) :
      {model}_{split}_auc.csv      tableau long complet (relu par load_results)
      {model}_{split}_summary.json params + resume (moyenne +/- std sur les seeds)"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for (model, split), d in df.groupby(["model", "split"]):
        tag = f"{model}_{split}"
        d.to_csv(out_dir / f"{tag}_auc.csv", index=False)

        run0 = next(r for r in runs if r.model.name == model and r.split == split)
        summary = {
            "model_name": model,
            "split": split,
            "seeds": sorted(d["seed"].unique().tolist()),
            "params": run0.model.params,
            "results": summarize(d).to_dict("records"),
        }
        with open(out_dir / f"{tag}_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=_json_default)
        print(f"Sauvegarde : {out_dir / (tag + '_auc.csv')} (+ summary.json)")


def load_results(out_dir=RESULTS_DIR) -> pd.DataFrame:
    """Relit tous les *_auc.csv de out_dir et les concatene : meme format que
    evaluate_runs. C'est la source des tableaux comparatifs et de l'EDA
    (plus aucune AUC ecrite a la main dans le code)."""
    files = sorted(Path(out_dir).glob("*_auc.csv"))
    if not files:
        raise FileNotFoundError(f"Aucun *_auc.csv dans {out_dir} : lancez save_results d'abord.")
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)


# ===================================================================== figure
def plot_auc_per_task(df, model, split, subset="test", out=None) -> Path:
    """Barres horizontales : AUC par tache (moyenne +/- std sur les seeds)."""
    d = df[(df["model"] == model) & (df["split"] == split) & (df["subset"] == subset)]
    if d.empty:
        raise ValueError(f"Aucun resultat pour {model}/{split}/{subset}")
    stats = d.groupby("task")["auc"].agg(["mean", "std"]).reindex(TASK_NAMES)
    stats = stats.dropna(subset=["mean"]).sort_values("mean")
    overall = stats["mean"].mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(stats.index, stats["mean"], xerr=stats["std"].fillna(0), color="#4C72B0")
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1, label="Hasard (0.5)")
    ax.axvline(overall, color="orange", linestyle="--", linewidth=1,
               label=f"Moyenne ({overall:.3f})")
    ax.set_xlim(0.4, 1)
    ax.set_xlabel(f"AUC ROC ({subset}, split {split})")
    ax.set_title(f"{model} - AUC par tache ({d['seed'].nunique()} seeds)")
    ax.legend()
    fig.tight_layout()

    out = Path(out) if out is not None else FIG_DIR / f"{model}_{split}_{subset}_auc_per_task.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Figure sauvegardee : {out}")
    return out