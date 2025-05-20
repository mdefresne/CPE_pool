import numpy as np
import os
import time
import pickle

from CPproblem import Roster, PCconfig, PCconfig_MO
from train import active_learning_loop
from pref_utils import plot_metrics, compute_metrics
from user import User
from TSPproblem import PC_TSP, PC_TSP_instance

import argparse


argparser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
argparser.add_argument("--pb", type=str, default="TSP", help="Problem name (PC or TSP)")
argparser.add_argument(
    "--relax",
    type=bool,
    default=True,
    help="Wether to train on a relaxed solution pool",
)
argparser.add_argument(
    "--update",
    type=str,
    default="MLE",
    help="Weight update ('SP', 'NCE', 'MLE' or 'MLE_online')",
)
argparser.add_argument(
    "--qs",
    type=str,
    default="cluster",
    help="Query selection ('cluster', 'ensemble', 'baseline' or 'baseline_relax')",
)
argparser.add_argument("--lr", type=float, default=1, help="learning rate")
argparser.add_argument("--alpha", type=float, default=4, help="gamma")
args = argparser.parse_args()

pb_name = args.pb
relax = args.relax
weight_update = [args.update]
QS = args.qs
eta = args.lr
alpha = args.alpha

if not os.path.exists("output/"):
    os.mkdir("output/")

# Init problem
sparse = False if pb_name == "TSP" else 0.8
diverse = False
n_users = 20
regret_step = 1

if pb_name == "PC":
    pb = PCconfig(mini=False)
    n_obj = pb.n_obj
    L_test_pb = [pb]
    alpha = 0.5

elif pb_name == "TSP":
    n_obj = 5
    size = 10
    pb = PC_TSP(size=size, n_obj=n_obj)
    relax = True
    n_test = 10  # num of test instances
    L_test_pb = [PC_TSP_instance(f"data_10{i}.txt", pb) for i in range(n_test)]


# Define training parameters
n_steps = 100
lbda_indif = 1  # large value for no indifference
lbda_pref = 1  # decrease for more noisy answer
val_set = [pb.random_solution() for _ in range(10)]
n_model = 25 if (QS == "ensemble" or QS == "cluster") else 1
n_epoch = 4
bs = 4
L1 = 0.5

# Metrics
total_train_time = 0
total_metrics = dict()
for update in weight_update:
    total_metrics[update] = {"val": [], "regret": [], "cs": [], "n_queries": []}

for seed in range(n_users):
    print("User", seed)
    user = User(
        seed,
        n_obj,
        sparse=sparse,
        lbda_pref=lbda_pref,
        lbda_indif=lbda_indif,
        problem_name=pb.name,
    )
    L_best_utility = [user.preferred_sol(test_pb)[1] for test_pb in L_test_pb]

    t0 = time.time()
    w, metrics, test_time = active_learning_loop(
        pb=pb,
        eta=eta,
        n_steps=n_steps,
        user=user,
        L_best_utility=L_best_utility,
        val_set=val_set,
        weight_update=weight_update,
        regret_step=regret_step,
        relax=relax,
        n_model=n_model,
        QS=QS,
        diverse=diverse,
        alpha=alpha,
        n_epoch=n_epoch,
        bs=bs,
        L1=L1,
        L_test_pb=L_test_pb,
    )
    total_train_time += time.time() - t0 - test_time
    for update in weight_update:
        for m in total_metrics[update].keys():
            total_metrics[update][m].append(np.array(metrics[update][m]))

print("Average training time: ", total_train_time / n_users)
metrics_std = dict()
metrics_mean = dict()
for update in weight_update:
    metrics_mean[update] = {"val": [], "regret": [], "cs": []}
    metrics_std[update] = {"val": [], "regret": [], "cs": []}
    for m in total_metrics[update].keys():
        metrics = np.array(total_metrics[update][m])
        metrics_mean[update][m] = np.mean(metrics, axis=0)
        metrics_std[update][m] = np.std(metrics, axis=0)

# Plotting figure
with open("output/metrics.pkl", "wb") as fp:
    pickle.dump(total_metrics, fp, protocol=pickle.HIGHEST_PROTOCOL)
plot_metrics(metrics_mean, pb.name + "_QS" + QS)  # , y_err=metrics_std)

# Printing metrics
for update in weight_update:
    print(f"### {update} ###")
    mu, se, cumul_reg, n_queries, user_sat, n_query_for_sat = compute_metrics(
        total_metrics, update
    )
    print(f"End regret: {mu} +- {se}")
    print(f"Cumulative regret : {np.round(cumul_reg, 2)}")
    print(f"Within 10% of best utility in {n_queries} queries")
    print(f"{user_sat*100}% of satisfied users")
    print(f"Among sat users, it took an average of {n_query_for_sat} queries")
