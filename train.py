import numpy as np
import math
import random
import cpmpy as cp
import time

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from scipy.stats import norm

from pref_utils import (
    utility_difference,
    regret,
    generate_feasible_sol,
    generate_random_sol,
)
from CPproblem import PCconfig
from TSPproblem import PC_TSP_instance


def active_learning_loop(
    pb,
    eta,
    n_steps,
    user,
    L_best_utility,
    val_set,
    weight_update=["MLE"],
    regret_step=1,
    relax=False,
    n_model=1,
    QS="random",
    diverse=False,
    alpha=0.5,
    n_epoch=4,
    bs=1,
    L1=1,
    L_test_pb=[],
):
    """
    Perform the full active learning training and validation.
    Input: - a CP problem pb
           - learning rate eta (float)
           - number of steps n_steps (int)
           - user
           - optimal utility value best_utility (float)
           - validation set val_set (list of solutions)
           - weight_update: list among "SP", "MLE_online", "MLE_indif", 'MLE', 'MLE_L1'
           - regret_step (int): compute regret every such steps (default is 1)
           - n_model (int): number of estimators to train (for ensemble-based queries)
           - QS (str): query selection method ('random' or 'ensemble')

    Output: - updated weights w (np.array)
            - tuple of 3 lists: the validation loss, the cosine similarity and the regret
    """

    if isinstance(weight_update, str):
        weight_update = [weight_update]

    w_init = np.ones((n_model, pb.n_obj))
    if n_model > 1:
        w_init += np.random.normal(0, 1, (n_model, pb.n_obj))
    weights = dict()
    metrics = dict()
    dataset = dict()
    selected_idx, selected_sol = dict(), dict()
    for update in weight_update:
        weights[update] = w_init.copy()
        metrics[update] = {"val": [], "regret": [], "cs": [], "n_queries": -1}
        dataset[update] = []
        selected_idx[update] = []
        selected_sol[update] = []
    L_reg = []

    ### Precomputing of the solution pool
    relax_sol = True if relax else False
    relax = False
    if relax_sol:
        use_fast_regret = False
        all_sol = generate_random_sol(pb, K=10000)
    else:  # generate feasible_solutions
        use_fast_regret = True
        K = 300 if diverse else 100000
        all_sol = generate_feasible_sol(pb, K, diverse)
    all_phi = [pb.features(sol) for sol in all_sol]

    if "TSP" in pb.name:
        PC_TSP_pb = pb

    for t in range(n_steps):
        if "TSP" in pb.name:
            pb = PC_TSP_instance(f"data_{t%50+1}.txt", PC_TSP_pb)
            # pb = PC_TSP_instance("data_1.txt", PC_TSP_pb)
            all_phi = [pb.features(sol) for sol in all_sol]

        if QS == "cluster":
            n_clusters = 5
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(
                all_phi
            )
            clusters = kmeans.labels_

        T = 0
        for idx, update in enumerate(weight_update):

            if QS == "random" and idx == 0:
                if relax:  # select random solution (not necessarily feasible)
                    y1 = pb.random_solution()
                    y2 = pb.random_solution()
                else:  # select random feasible
                    chosen_idx = np.random.choice(len(all_sol), 2)
                    y1, y2 = all_sol[chosen_idx]

            elif QS == "ensemble" or QS == "cluster":
                if alpha <= 1:
                    a = alpha
                elif alpha == 2:
                    a = t / n_steps
                elif alpha == 3:
                    a = 1 - 1 / (t + 1) ** 0.5
                elif alpha == 4:
                    a = 1 - 1 / (t + 1)

                if QS == "ensemble":
                    y1, y2 = ensembleQS(
                        weights[update],
                        all_phi,
                        all_sol,
                        selected_idx[update],
                        alpha=a,
                        num_sol=2,
                    )

                elif QS == "cluster":
                    y1, y2 = QS_cluster(
                        clusters,
                        all_sol,
                        weights[update],
                        all_phi,
                        alpha=a,
                        num_sol=2,
                        QS="ensemble",
                    )

            elif QS == "CSS":
                y1, y2 = CSS(
                    pb, weights[update][0], selected_sol[update], relax, num_sol=2
                )
                selected_sol[update] += [y1, y2]

            elif QS == "baseline":
                y1, y2 = QS_choicePerceptron(
                    pb, weights[update][0], selected_sol[update], gamma=1 / (t + 1)
                )
                selected_sol[update] += [y1, y2]

            elif QS == "baseline_relax":
                y1, y2 = choicePerceptron_relaxed(
                    pb, all_phi, all_sol, weights[update][0], gamma=1 / (t + 1)
                )

            # compute features
            phi1 = pb.features(y1)
            phi2 = pb.features(y2)

            # oracle to get the preferred one
            answer = user.answer(user._utility_diff(phi1, phi2))
            is_user_sat = metrics[update]["n_queries"] != -1
            if (not is_user_sat) and (answer != 0):
                # user has a clear preference and not yet satisfied
                if answer == 1:  # y1 is preferred
                    phi_plus, phi_minus = phi1, phi2
                else:
                    phi_plus, phi_minus = phi2, phi1
                delta = phi_plus - phi_minus
                dataset[update].append([phi_plus, phi_minus])

                # update weight
                util1 = np.dot(weights[update], phi_plus)
                util2 = np.dot(weights[update], phi_minus)
                is_pred_wrong = util1 < util2

                if update == "SP":
                    if is_pred_wrong.any():
                        weights[update][is_pred_wrong] += eta * delta
                elif update == "NCE":
                    # if t % 3 == 0 and len(dataset[update]) > 0 and QS == "baseline":
                    #   eta = cross_validate(dataset[update], L_eta, w_init)
                    weights[update] += eta * delta
                elif (update == "MLE_online") or (update == "MLE_indif"):
                    grad = grad_MLE(weights[update], delta)
                    weights[update] -= eta * grad
                elif update in ["MLE", "MLE_L1"]:
                    L1_coef = L1 if update == "MLE_L1" else 0
                    weights[update] = train_MLE(
                        dataset[update],
                        eta,
                        w_init,
                        n_epoch=n_epoch,
                        bs=bs,
                        L1=L1_coef,
                    )

            else:  # indifference
                if update == "MLE_indif":
                    grad = 0.5 * grad_MLE(
                        weights[update], phi1 - phi2
                    ) + 0.5 * grad_MLE(weights[update], phi2 - phi1)
                    weights[update] -= eta * grad

            # store metrics (if indifference, they will be the same as in previous step)
            cs = np.mean(
                [
                    cosine_similarity(w.reshape(1, -1), user.w.reshape(1, -1))[0, 0]
                    for w in weights[update]
                ]
            )
            val_regret = np.mean(
                [
                    np.mean([utility_difference(y, w, user.w, pb) for y in val_set])
                    for w in weights[update]
                ]
            )
            metrics[update]["val"].append(val_regret)
            metrics[update]["cs"].append(cs)

            if (t == n_steps - 1) or (t % regret_step == 0):
                t0 = time.time()
                pred_w = np.mean(weights[update], axis=0)
                L_reg, L_absolute_reg = [], []
                for i, best_utility in enumerate(L_best_utility):
                    test_pb = L_test_pb[i]
                    if use_fast_regret:  # requires all feasible solutions
                        reg = fast_regret(pred_w, all_phi, best_utility, user.w)
                    else:
                        reg = regret(best_utility, pred_w, user.w, test_pb, relax=relax)
                    L_reg.append(reg)
                    L_absolute_reg.append(reg * abs(best_utility))
                metrics[update]["regret"].append(np.mean(L_reg))

                # check if user is satisfied (ie indifferent between top and current solution)
                answer = user.answer(np.mean(L_absolute_reg), val=True)
                print("Regret", np.mean(L_reg), answer)
                T += time.time() - t0
                if not is_user_sat and (answer == 0):
                    print("User satisfied")
                    metrics[update]["n_queries"] = t

    #print("Regret", np.mean(L_reg), answer)
    return weights, metrics, T


def grad_MLE(w, delta, lbda=1):

    factor = 1 / (1 + np.exp(np.dot(w, delta) / lbda))
    delta = np.repeat(delta.reshape(1, -1), w.shape[0], axis=0)

    return -delta * factor.reshape(-1, 1)


def train_MLE(dataset, eta, w_init, n_epoch=1, bs=1, L1=0):
    """
    Train with MLE over the whole dataset
    Dataset is a list of (phi_plus, phi_minus)
    """
    w = w_init.copy()
    eta /= n_epoch  # to have the amplitude of updates as other methods
    reg_term = L1 / len(dataset)

    for _ in range(n_epoch):
        shuffle_idx = np.random.choice(
            np.arange(len(dataset)), len(dataset), replace=False
        )
        grad = 0
        for batch_idx, sample_idx in enumerate(shuffle_idx):
            phi_plus, phi_minus = dataset[sample_idx]
            delta = phi_plus - phi_minus
            grad += grad_MLE(w, delta)  # * 1 / bs
            if L1 > 0:
                grad += reg_term * np.sign(w).squeeze()
            if batch_idx % bs == 0:
                w -= eta * grad
                grad = 0
        w -= eta * grad  # LR should be prop to bs (or sqrt bs)

    return w


def train_SP(dataset, eta, w_init):
    for i in range(len(dataset)):
        phi_plus, phi_minus = dataset[i]
        w = w_init.copy()
        w += eta * (phi_plus - phi_minus)

    return w


def cross_validate(dataset, L_eta, w_init):

    dataset = np.array(dataset)  # n*2*n_ft
    perfs, correct = [], []
    for eta in L_eta:
        w = train_SP(dataset, eta, w_init)
        Delta = dataset[:, 0] - dataset[:, 1]
        estim_util = np.dot(Delta, w.squeeze())
        acc = np.sum(estim_util[estim_util > 0])
        perfs.append(acc.mean())
        correct.append(np.mean(estim_util > 0))

    perfs = np.array(perfs)
    # perfs[correct < np.max(correct)] = 0
    best_eta = np.argmax(perfs)
    return L_eta[best_eta]


def fast_regret(pred_w, all_phi, best_utility, true_w):

    all_utilities = np.dot(all_phi, pred_w)  # n_sol, n_model
    best_idx = np.argsort(all_utilities)[::-1][0]
    true_utility = np.dot(all_phi[best_idx], true_w)
    reg = abs(true_utility - best_utility) / best_utility
    return reg


def ensembleQS(
    current_w,
    all_phi,
    feasible_sol,
    selected_idx=[],
    alpha=0.5,
    num_sol=2,
):
    """
    Performs query selection based on an ensemble of models.
    Alpha is a trade-off between exploitation and exploration (alpha * mean + (1-alpha) * var)
    """

    all_utilities = np.dot(all_phi, current_w.T)  # n_sol, n_model
    sigma = np.var(all_utilities, axis=1)
    mu = np.mean(all_utilities, axis=1)
    acquisition = alpha * mu + (1 - alpha) * sigma
    # sorted_idx = np.argsort(np.max(all_utilities, axis=1))[::-1]
    sorted_idx = np.argsort(acquisition)[::-1]

    L_idx, i = [], 0
    while len(L_idx) < num_sol:
        top_idx = sorted_idx[i]
        if top_idx not in selected_idx:
            L_idx.append(top_idx)
            selected_idx.append(top_idx)
        i += 1

    # print(clusters[L_idx[0]], clusters[L_idx[1]])
    return feasible_sol[L_idx]


def QS_cluster(
    clusters, feasible_sol, current_w, all_phi, alpha=0.5, num_sol=2, QS="random"
):

    N_sol = len(clusters)
    n_clusters = np.max(clusters)
    # cluster_idx = np.random.choice(np.arange(n_clusters), size=2, replace=False)
    cluster_idx = np.random.randint(0, n_clusters, num_sol)

    L_query = []
    for c_idx in cluster_idx:
        cluster_sol = feasible_sol[clusters == c_idx]
        if QS == "ensemble":
            cluster_phi = np.array(all_phi)[clusters == c_idx]
            L_query = ensembleQS(
                current_w,
                cluster_phi,
                cluster_sol,
                selected_idx=[],
                alpha=alpha,
                num_sol=num_sol,
            )
        else:
            query = cluster_sol[np.random.choice(len(cluster_sol))]
            L_query.append(query)

    return L_query


def CSS(pb, current_w, selected_sol, relax=False, num_sol=2):
    """Current solution strategy (best vs 2nd best)"""

    model = pb.relaxed_model if relax else pb.model
    s = cp.SolverLookup.get("ortools", model)
    for sol in selected_sol:  # different solutions than previous
        s += sum(pb.variables != sol) > 0

    solutions = []
    for _ in range(num_sol):
        s.maximize(pb.objective(pb.variables, current_w.squeeze()))
        s.solve()
        solutions.append(pb.variables.value())
        s += sum(pb.variables != pb.variables.value()) > 0

    return solutions


def QS_choicePerceptron(pb, current_w, selected_sol, gamma):
    if pb.name == "PC":
        y1, y2 = QS_choicePerceptron_PC(current_w, selected_sol, gamma)
    if "TSP" in pb.name:
        # selected_sol = []  # multiple instances
        y1, y2 = QS_choicePerceptron_TSP(pb, current_w, selected_sol, gamma)
    return y1, y2


def QS_choicePerceptron_PC(current_w, selected_sol, gamma):

    pb = PCconfig(mini=False)
    s = cp.SolverLookup.get("ortools", pb.model)
    s.maximize(pb.objective(pb.variables, current_w.squeeze()))
    for sol in selected_sol:
        s += cp.any(pb.variables != sol)
    s.solve()
    y1 = pb.variables.value().copy()

    pb = PCconfig(mini=False, int_cost=True)
    sobj1, norm = pb.sub_objectives(y1)
    sobj2, _ = pb.sub_objectives(pb.variables)

    max_price = int(1 / norm[-1])
    sobj1[:-1] *= max_price
    sobj2[:-1] *= max_price
    delta = cp.sum(abs(sobj2 - sobj1))
    mu = np.dot(current_w, sobj2)

    pb.model.maximize(gamma * delta + (1 - gamma) * mu)
    pb.model += cp.any(pb.variables != y1)
    for sol in selected_sol:
        # different solutions than previous (otherwise stuck i correct prediction)
        pb.model += cp.any(pb.variables != sol)
    pb.model.solve()
    y2 = pb.variables.value().copy()

    return y1, y2


def QS_choicePerceptron_TSP(pb, current_w, selected_sol, gamma):

    y1, _ = pb.solve(current_w, QSinfo=(selected_sol, None, None, None))
    phi1 = pb.features(y1)
    y2, _ = pb.solve(current_w, QSinfo=(selected_sol, y1, phi1, gamma))

    return y1, y2


def choicePerceptron_relaxed(pb, all_phi, all_sol, current_w, gamma):

    all_utilities = np.dot(all_phi, current_w.T)  # n_sol, n_model
    idx1 = np.argmax(all_utilities)
    phi1 = all_phi[idx1]
    delta = np.sum(abs(all_phi - phi1), axis=1)
    acquisition = gamma * delta + (1 - gamma) * all_utilities
    sorted_idx = np.argsort(acquisition)[::-1]
    idx2 = sorted_idx[0] if sorted_idx[0] != idx1 else sorted_idx[1]

    return all_sol[idx1], all_sol[idx2]
