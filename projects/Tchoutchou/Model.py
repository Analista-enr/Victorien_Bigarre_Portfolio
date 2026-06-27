from pyswarm import pso
from scipy.optimize import basinhopping
import numpy as np
import pandas as pd
import time
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Union
from ortools.sat.python import cp_model
from pulp import LpProblem, LpVariable, LpMinimize, LpBinary, LpStatus, lpSum
from deap import base, creator, tools, algorithms

class TchoutchouModel:
    def __init__(self, df: pd.DataFrame,nb_quai : int):
        # Tri par numéro (descendant)
        self.df = df.sort_values(by=["Numero"], ascending=False)

        # Paramètres
        self.T = 24 * 60
        self.nb_quais = nb_quai
        self.nb_train = len(self.df)
        self.penalty = 10


        self.A = [
            [np.zeros(self.T, dtype=int) for _ in range(self.nb_quais)]
            for _ in range(self.nb_train)
        ]

        self.dico = self.get_train_schedule()

    def get_train_schedule(self) -> Dict:
        """Retourne un dictionnaire avec les horaires et l'emplacement de chaque train."""
        dico = {}
        for num in self.df["Numero"]:
            train_data = {}
            # Extraction des heures/minutes (suppose que Arrival_real et Departure_real sont des datetime64)
            arrival = pd.to_datetime(self.df.loc[self.df["Numero"] == num, "Arrival_real"].iloc[0])
            departure = pd.to_datetime(self.df.loc[self.df["Numero"] == num, "Departure_real"].iloc[0])

            train_data["arr"] = [arrival.hour, arrival.minute]
            train_data["dep"] = [departure.hour, departure.minute]
            train_data["emp"] = self.df[self.df["Numero"] == num].index[0]  # Emplacement (index)
            dico[num] = train_data
        return dico

    def op_solution(self, N: List[int]) -> List:
        """Génère une solution à partir d'une affectation N (liste d'index de quais)."""
        solution = [row[:] for row in self.A]
        for num in self.dico:
            deb = self.dico[num]["arr"][0] * 60 + self.dico[num]["arr"][1]
            fin = self.dico[num]["dep"][0] * 60 + self.dico[num]["dep"][1]
            index = self.dico[num]["emp"]
            quai = int(round(N[index],0))
            solution[index][quai] = np.array([
                1 if (t >= deb) and (t <= fin + 1) else 0
                for t in range(self.T)
            ])
        return solution

    def dist(self, A: List) -> float:
        """Calcule la distance entre les trains (chevauchements)."""
        dist = 0
        for j in range(self.nb_quais):
            s = 0
            for i in range(self.nb_train):
                for k in range(i + 1, self.nb_train):

                    s += np.dot(A[i][j], A[k][j])
            dist += s
        return dist

    def op_score(self, A: List) -> float:
        """Calcule le score de la solution (pénalité * distance)."""
        return self.dist(A) * self.penalty

    def _run_pso(self, n_particles=50, max_iter=100, w=0.7, c1=1.5, c2=1.5):
        """PSO avec pyswarm et arrêt si score=0."""
        lb, ub = np.zeros(self.nb_train), np.full(self.nb_train, self.nb_quais - 1)
        self._best_zero_solution = None  # Réinitialisation

        def objective(x):
            N = np.clip(np.round(x), 0, self.nb_quais - 1).astype(int).tolist()
            score = self.op_score(self.op_solution(N))
            if score == 0:
                self._best_zero_solution = N  # Stocke la solution parfaite
            return score

        best_x, best_score = pso(
            objective, lb, ub, swarmsize=n_particles, maxiter=max_iter
        )

        # Si une solution parfaite a été trouvée pendant l'optimisation
        if self._best_zero_solution is not None:
            return self._best_zero_solution, 0.0

        best_N = np.clip(np.round(best_x), 0, self.nb_quais - 1).astype(int).tolist()
        return best_N, best_score

    def _run_genetic(self, pop_size=50, ngen=100, cx_prob=0.5, mut_prob=0.2):
        """Algorithme Génétique avec DEAP et arrêt si score=0."""
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        self._best_zero_solution = None  # Réinitialisation

        toolbox = base.Toolbox()
        toolbox.register("attr_quai", np.random.randint, 0, self.nb_quais)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                         toolbox.attr_quai, n=self.nb_train)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def evaluate(ind):
            score = self.op_score(self.op_solution(ind))
            if score == 0:
                self._best_zero_solution = ind  # Stocke la solution parfaite
            return (score,)

        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
        toolbox.register("mutate", tools.mutUniformInt, low=0, up=self.nb_quais - 1, indpb=mut_prob)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)

        for gen in range(ngen):
            algorithms.varAnd(pop, toolbox, cxpb=cx_prob, mutpb=mut_prob)
            fits = toolbox.map(toolbox.evaluate, pop)
            for ind, fit in zip(pop, fits):
                ind.fitness.values = fit
            hof.update(pop)
            if self._best_zero_solution is not None:  # Arrêt si score=0 trouvé
                return self._best_zero_solution, 0.0

        return hof[0], hof[0].fitness.values[0]

    def _run_sa(self, niter=1000, stepsize=0.5, T=1.0):
        """Recuit Simulé avec basinhopping et arrêt si score=0."""
        self._best_zero_solution = None  # Réinitialisation

        def objective(N):
            score = self.op_score(self.op_solution(N))
            if score == 0:
                self._best_zero_solution = N  # Stocke la solution parfaite
            return score

        N0 = np.random.randint(0, self.nb_quais, self.nb_train).tolist()

        def callback(x, f, accept):
            if f == 0:
                self._best_zero_solution = x  # Stocke la solution parfaite
                return True  # Arrête basinhopping
            return False

        result = basinhopping(
            objective, x0=N0, niter=niter, stepsize=stepsize, T=T,
            minimizer_kwargs={"method": "L-BFGS-B",
                              "bounds": [(0, self.nb_quais - 1) for _ in range(self.nb_train)]},
            callback=callback
        )

        if self._best_zero_solution is not None:  # Si arrêt prématuré
            return self._best_zero_solution, 0.0
        return result.x, result.fun

    # ====================== HYBRIDE PSO+SA ======================
    def _run_pso_sa(self, n_particles=50, max_iter=100, sa_niter=50, sa_stepsize=0.5):
        """Hybride PSO + Recuit Simulé (minimisation)."""
        # Étape 1 : PSO pour obtenir une solution initiale
        pso_N, pso_score = self._run_pso(n_particles=n_particles, max_iter=max_iter)

        # Si le PSO a déjà trouvé une solution parfaite (score=0), on retourne directement
        if pso_score == 0:
            return pso_N, 0.0

        # Étape 2 : Recuit Simulé (implémentation manuelle pour plus de contrôle)
        current_solution = pso_N.copy()
        current_score = pso_score
        best_solution = current_solution.copy()
        best_score = current_score
        temp = 1.0
        cooling_rate = 0.99

        # Fonction pour générer un voisin (swap aléatoire de 2 trains)
        def generate_neighbor( solution):
            neighbor = solution.copy()
            move_type = random.choice(['swap', 'move', 'reverse'])
            if move_type == 'swap':
                i, j = random.sample(range(len(neighbor)), 2)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            elif move_type == 'move':
                i = random.randint(0, len(neighbor) - 1)
                neighbor[i] = random.randint(0, self.nb_quais - 1)
            elif move_type == 'reverse':
                i, j = sorted(random.sample(range(len(neighbor)), 2))
                neighbor[i:j + 1] = neighbor[i:j + 1][::-1]
            return neighbor

        for _ in range(sa_niter):
            # Génère un voisin
            new_solution = generate_neighbor(current_solution)
            new_score = self.op_score(self.op_solution(new_solution))

            # Si on trouve une solution parfaite, on stocke et on arrête
            if new_score == 0:
                return new_solution, 0.0

            # Accepte si meilleur ou avec probabilité (SA)
            delta = new_score - current_score
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_solution, current_score = new_solution, new_score
                if current_score < best_score:
                    best_solution, best_score = current_solution, current_score
            temp *= cooling_rate

        return best_solution, best_score

    def _run_pulp(self) -> Tuple[List[int], float]:
        """PuLP (MILP) pour une solution optimale ou relaxée."""
        prob = LpProblem("Train_Quai_Assignment", LpMinimize)
        x = [[LpVariable(f'x_{i}_{j}', cat=LpBinary) for j in range(self.nb_quais)] for i in range(self.nb_train)]

        # Contrainte: chaque train est assigné à exactement un quai
        for i in range(self.nb_train):
            prob += lpSum(x[i][j] for j in range(self.nb_quais)) == 1

        # Pré-calculer les chevauchements entre paires de trains
        overlap = [[0] * self.nb_train for _ in range(self.nb_train)]
        for i in self.dico:
            for k in self.dico:
                if i >= k:
                    continue
                deb_i = self.dico[i]["arr"][0] * 60 + self.dico[i]["arr"][1]
                fin_i = self.dico[i]["dep"][0] * 60 + self.dico[i]["dep"][1]
                deb_k = self.dico[k]["arr"][0] * 60 + self.dico[k]["arr"][1]
                fin_k = self.dico[k]["dep"][0] * 60 + self.dico[k]["dep"][1]
                overlap[[j for j in self.dico].index(i)][[j for j in self.dico].index(k)] = max(0, min(fin_i, fin_k) - max(deb_i, deb_k))

        # Variable auxiliaire pour les chevauchements
        w = {}
        total_overlap = 0
        for j in range(self.nb_quais):
            for i in range(self.nb_train):
                for k in range(i + 1, self.nb_train):
                    if overlap[i][k] > 0:
                        w[i, k, j] = LpVariable(f'w_{i}_{k}_{j}', cat=LpBinary)
                        prob += w[i, k, j] >= x[i][j] + x[k][j] - 1
                        prob += w[i, k, j] <= x[i][j]
                        prob += w[i, k, j] <= x[k][j]
                        total_overlap += overlap[i][k] * w[i, k, j]

        prob += total_overlap
        prob.solve()

        if LpStatus[prob.status] in ('Optimal', 'Feasible'):
            solution = []
            for i in range(self.nb_train):
                for j in range(self.nb_quais):
                    if x[i][j].varValue == 1:
                        solution.append(j)
                        break
            return solution, float(self.op_score(self.op_solution(solution)))
        else:
            random_solution = np.random.randint(0, self.nb_quais, self.nb_train).tolist()
            return random_solution, float(self.op_score(self.op_solution(random_solution)))

    # ====================== COMPARATIF ======================
    def compare_optimizers(self, n_iter=100, pop_size=50, n_particles=50,
                           sa_niter=1000, sa_stepsize=0.5, verbose=False):
        """
        Compare PSO, Génétique, Recuit Simulé et PSO+SA.
        Args:
            n_iter: Itérations pour PSO/GA.
            pop_size: Taille de la population pour GA.
            n_particles: Nombre de particules pour PSO.
            sa_niter: Itérations pour SA et PSO+SA.
            sa_stepsize: Taille des pas pour SA.
            verbose: Affiche les résultats.
        Returns:
            Dict avec les résultats des 4 algorithmes.
        """
        results = {}

        # --- PSO ---
        print("### === PSO running ..... === ###")
        start = time.time()
        pso_N, pso_score = self._run_pso(n_particles=n_particles, max_iter=n_iter)
        pso_time = time.time() - start
        results['pso'] = {
            'N': list(pso_N),
            'score': float(pso_score),
            'time': float(pso_time)
        }
        print("### === PSO done ! === ###")

        # --- MILP ---
        start = time.time()
        print("### === MILP running ..... === ###")
        milp_N, milp_score = self._run_pulp()
        milp_time = time.time() - start
        results['milp'] = {
            'N': list(milp_N),
            'score': float(milp_score),
            'time': float(milp_time)
        }
        print("### === MILP done ! === ###")

        # --- Génétique ---
        start = time.time()
        print("### === GA running ..... === ###")
        ga_N, ga_score = self._run_genetic(pop_size=pop_size, ngen=n_iter)
        ga_time = time.time() - start
        results['genetic'] = {
            'N': list(ga_N),
            'score': float(ga_score),
            'time': float(ga_time)
        }
        print("### === GA done ! === ###")

        # --- Recuit Simulé ---
        start = time.time()
        print("### === SA running ..... === ###")
        sa_N, sa_score = self._run_sa(niter=sa_niter, stepsize=sa_stepsize)
        sa_time = time.time() - start
        results['simulated_annealing'] = {
            'N': list(sa_N),
            'score': float(sa_score),
            'time': float(sa_time)
        }
        print("### === SA done ! === ###")

        # --- PSO+SA ---
        start = time.time()
        print("### === Hybrid running ..... === ###")
        pso_sa_N, pso_sa_score = self._run_pso_sa(
            n_particles=n_particles,
            max_iter=n_iter,
            sa_niter=sa_niter,
            sa_stepsize=sa_stepsize
        )
        pso_sa_time = time.time() - start
        results['pso_sa'] = {
            'N': list(pso_sa_N),
            'score': float(pso_sa_score),
            'time': float(pso_sa_time)
        }
        print("### === SA done ! === ###")

        if verbose:
            print("\n=== COMPARATIF ===")
            for name, res in results.items():
                print(f"{name.upper():<18} | Score: {res['score']:.2f} | Temps: {res['time']:.3f}s")

        return results

    def plot_all_results(self, results: Dict[str, List[Dict]]):
        """Génère 6 graphiques pour analyser les performances."""
        methods = list(results.keys())
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 3)

        # 1. Boxplot des scores
        ax1 = fig.add_subplot(gs[0, 0])
        scores_data = []
        for m in methods:
            scores_data.extend([r['score'] for r in results[m]])
        sns.boxplot(x=[m for m in methods for _ in range(len(results[m]))],
                    y=scores_data, ax=ax1, palette="Set2")
        ax1.set_title("📊 Distribution des scores par algorithme", fontsize=12)
        ax1.set_ylabel("Score (chevauchements × pénalité)")
        ax1.tick_params(axis='x', rotation=45)

        # 2. Temps moyen par algorithme
        ax2 = fig.add_subplot(gs[0, 1])
        times_mean = [np.mean([r['time'] for r in results[m]]) for m in methods]
        times_std = [np.std([r['time'] for r in results[m]]) for m in methods]
        ax2.bar(methods, times_mean, yerr=times_std, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax2.set_title("⏱️ Temps d'exécution moyen (s)", fontsize=12)
        ax2.set_ylabel("Temps (s)")
        ax2.grid(True, alpha=0.3)

        # 3. Heatmap des affectations (1er run)
        ax3 = fig.add_subplot(gs[0, 2])
        first_solutions = [results[m][0]['N'] for m in methods]
        heatmap_data = np.zeros((self.nb_quais, self.nb_train))
        for sol in first_solutions:
            for train_idx, quai in enumerate(sol):
                heatmap_data[quai][train_idx] += 1
        sns.heatmap(heatmap_data, ax=ax3, cmap="YlGnBu", xticklabels=10, yticklabels=True)
        ax3.set_title("🔥 Heatmap des affectations (1er run)", fontsize=12)
        ax3.set_xlabel("Train")
        ax3.set_ylabel("Quai")

        # 4. Quai assigné par train (tous les runs)
        ax4 = fig.add_subplot(gs[1, :])
        for m in methods:
            for run in results[m]:
                ax4.plot(run['N'], 'o-', label=f"{m} (run {results[m].index(run)+1})", linewidth=1, markersize=3, alpha=0.7)
        ax4.axhline(y=self.nb_quais-1, color='red', linestyle='--', alpha=0.5, label=f"Max ({self.nb_quais-1})")
        ax4.set_title("📈 Quai assigné par train (tous les runs)", fontsize=12)
        ax4.set_xlabel("Train")
        ax4.set_ylabel("Quai")
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)

        # 5. Histogramme des scores
        ax5 = fig.add_subplot(gs[2, 0])
        for m in methods:
            ax5.hist([r['score'] for r in results[m]], alpha=0.5, label=m, bins=10)
        ax5.set_title("📉 Histogramme des scores", fontsize=12)
        ax5.set_xlabel("Score")
        ax5.set_ylabel("Fréquence")
        ax5.legend()

        # 6. Temps vs Score (scatter)
        ax6 = fig.add_subplot(gs[2, 1])
        for m in methods:
            times = [r['time'] for r in results[m]]
            scores = [r['score'] for r in results[m]]
            ax6.scatter(times, scores, label=m, alpha=0.6, s=50)
        ax6.set_title("⚡ Temps vs Score", fontsize=12)
        ax6.set_xlabel("Temps (s)")
        ax6.set_ylabel("Score")
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # 7. Boîte à moustaches pour le temps
        ax7 = fig.add_subplot(gs[2, 2])
        times_data = []
        for m in methods:
            times_data.extend([r['time'] for r in results[m]])
        sns.boxplot(x=[m for m in methods for _ in range(len(results[m]))],
                    y=times_data, ax=ax7, palette="Set2")
        ax7.set_title("⏳ Distribution des temps par algorithme", fontsize=12)
        ax7.set_ylabel("Temps (s)")
        ax7.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()
