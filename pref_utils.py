import numpy as np
import cpmpy as cp
import math
from matplotlib import pyplot as plt
from os import listdir


# evaluation criterion
def utility_difference(y, w, true_w, pb, rescale=True):
    """
    Compute the difference in utility with estimate w instead of the true utility.
    Rescale the predicted weight in the same range as the true weights.
    it's not the true regret but it's much faster to compute as no solving is involved.
    """

    true_util = pb.objective(y, true_w)
    if rescale:
        w = w * max(abs(true_w)) / max(abs(w))
    estim_util = pb.objective(y, w)

    return abs(true_util - estim_util)


def regret(optim_util, pred_w, true_w, pb, relax=False, y=None):
    """
    Compute the regret as the difference between the true utility of the optimal solution
    vs the true utility of the estimated solution.
    if relax, solve the relaxed problem.
    Return regret as a % of optim util
    """
    if y is None:
        pred_sol, _ = pb.solve(pred_w, relax)
    else:
        pred_sol = y

    phi = pb.features(pred_sol)
    real_util_pred_sol = np.dot(true_w, phi)
    regret = optim_util - real_util_pred_sol

    return regret / abs(optim_util)


def plot_metrics(metrics, figname="Mock", y_err=None):

    plt.figure(figsize=(20, 5))
    for update in metrics.keys():
        L_val = metrics[update]["val"]
        L_cs = metrics[update]["cs"]
        L_regret = metrics[update]["regret"]
        if len(L_regret) == 1:
            regret_step = len(L_val)
        else:
            regret_step = len(L_val) // (len(L_regret) - 1)
        L_regret = np.repeat(L_regret[:-1], regret_step)
        L_regret = np.concatenate((L_regret[:-1], np.array([L_regret[-1]])))
        X = np.arange(len(L_val))

        plt.subplot(131)
        yerr = y_err[update]["val"] if y_err is not None else None
        plt.errorbar(X, L_val, yerr=yerr, label=update)
        plt.legend()
        plt.xlabel("Number of queries")
        plt.ylabel("Validation loss")
        # Plateau because no update if used indifferent

        plt.subplot(132)
        yerr = y_err[update]["cs"] if y_err is not None else None
        plt.errorbar(X, L_cs, yerr=yerr, label=update)
        plt.legend()
        plt.xlabel("Number of queries")
        plt.ylabel("Cosine similarity")

        plt.subplot(133)
        X = np.arange(len(L_regret))
        yerr = (
            np.repeat(y_err[update]["regret"], regret_step)
            if y_err is not None
            else None
        )
        plt.errorbar(X, L_regret, yerr=yerr, label=update)
        plt.xlabel("Number of queries")
        plt.ylabel("Relative regret (%)")
        plt.plot(X, [0.1] * len(X), "k--")
        plt.legend()

    plt.savefig("output/" + figname, bbox_inches="tight")


def generate_feasible_sol(pb, K=100, diverse=False):
    """Generate a list of feasible solutions, optionally diverse."""

    store = []
    filename = (
        f"sol-diverse_{pb.name}_K{K}.npy"
        if diverse
        else f"sol-feasible_{pb.name}_K{K}.npy"
    )
    if filename not in listdir("output/"):
        if diverse:
            s = cp.SolverLookup.get(
                "ortools", pb.model
            )  # faster on a solver interface directly

            while len(store) < K and s.solve():
                print(len(store), ":", pb.variables.value(), s.objective_value())
                store.append(pb.variables.value())
                # Hamming dist: nr of different elements + penalty if reuse same sol
                s.maximize(sum([sum(pb.variables != sol) for sol in store]))
                s += sum(pb.variables != pb.variables.value()) > 0

            np.save("output/" + filename, store)

        else:
            pb.model.maximize(None)
            n_sol = pb.model.solveAll(
                solution_limit=K,
                display=lambda: store.append(list(pb.variables.value())),
            )
            np.save("output/" + filename, store)

    else:
        store = np.load("output/" + filename)

    return np.array(store)


def generate_random_sol(pb, K=1000):
    """Generate a list of random solutions."""

    store = []
    filename = f"sol-random_{pb.name}_K{K}.npy"
    if filename not in listdir("output/"):
        store = [pb.random_solution()]
        while len(store) < K:
            y = pb.random_solution()
            if not np.any(np.all(np.array(store) == y, axis=1)):
                store.append(y)
        np.save("output/" + filename, store)

    else:
        store = np.load("output/" + filename)

    return np.array(store)


def compute_metrics(metrics, update):

    L_val = metrics[update]["val"]
    L_regret = np.array(metrics[update]["regret"]) * 100  # in %
    L_queries = np.array(metrics[update]["n_queries"])
    n_users, n_regret = L_regret.shape
    if n_regret > 1:
        n_regret -= 1  # because last step regret
    regret_step = len(L_val[0]) // n_regret
    avg_regret = np.mean(L_regret, axis=0)
    mu = np.round(avg_regret[-1], 3)
    sigma = np.round(np.std(L_regret[:, -1]), 3)
    se = sigma / n_users
    cumul_reg = np.sum(
        avg_regret
        * np.array([1 / (t * n_regret) for t in range(1, len(avg_regret) + 1)])
    )

    if np.min(avg_regret) < 10:
        n_queries = np.argmax(avg_regret < 10) * regret_step
    else:
        n_queries = -1

    user_sat = np.mean(L_queries > -1)
    n_query_for_sat = np.mean(L_queries[L_queries > -1])

    return mu, se, cumul_reg, n_queries, user_sat, n_query_for_sat
