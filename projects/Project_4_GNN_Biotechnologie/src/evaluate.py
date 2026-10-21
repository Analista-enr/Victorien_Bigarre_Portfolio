self.PREDICTORS = {
    "SVM_Linear": self._predict_svm_linear,
    "SVM_Tanimoto": self._predict_svm_tanimoto,
    "LGBM": self._predict_lgbm,
}
def _predict_svm_linear(
        self,
        clf,
        Xte,
        G_te,
):
    """Marge du SVM lineaire (ranking pour l'AUC, pas besoin de proba)."""
    return clf.decision_function(Xte)

def _predict_svm_tanimoto(self, clf, Xte, G_te):
    """Marge du SVM Tanimoto : G_te = Gram (test masque x train masque)."""
    return clf.decision_function(G_te)

def _predict_lgbm(self, clf, Xte, G_te):
    """Probabilite de la classe positive LGBM (probas natives)."""
    return clf.predict_proba(Xte)[:, 1]

def evaluate(self, model_name, models, idx_train, idx_test) -> dict:
    """Pipeline d'evaluation unique, miroir de training().
    Retourne {"auc_per_task": (12,), "auc_mean": float}.

    Invariant cle : le masque train de chaque tache est reconstruit ici
    par la MEME formule que dans training() — c'est ce qui garantit que
    la Gram croisee a la largeur attendue par le modele Tanimoto.
    """
    if model_name not in self.PREDICTORS:
        raise ValueError(
            f"Modele inconnu : {model_name!r}. "
            f"Choix valides : {list(self.PREDICTORS)}"
        )

    X, Y = self.DATA.features, self.DATA.labels
    G = self.DATA.gram_tanimoto
    predict = self.PREDICTORS[model_name]

    aucs = []
    for t, clf in enumerate(models):
        # test masque (lignes de la metrique)
        rows_te = idx_test[~np.isnan(Y[idx_test, t])]
        yte = Y[rows_te, t].astype(int)

        # garde-fou : AUC indefinie sans deux classes
        if len(np.unique(yte)) < 2:
            aucs.append(np.nan)
            continue

        # train masque : LA meme formule que training() (invariant de largeur)
        rows_tr = idx_train[~np.isnan(Y[idx_train, t])]
        G_te = G[np.ix_(rows_te, rows_tr)]  # Gram croisee test x train
        scores = predict(clf=clf, Xte=X[rows_te], G_te=G_te)
        aucs.append(roc_auc_score(yte, scores))

    aucs = np.array(aucs)
    return {"auc_per_task": aucs, "auc_mean": np.nanmean(aucs)}
