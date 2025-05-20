import numpy as np
import os
import pickle
import pandas as pd

from CPproblem import Roster, PCconfig, PCconfig_MO
from train import active_learning_loop
from pref_utils import plot_metrics, compute_metrics
from user import User


if not os.path.exists("output/"):
    os.mkdir("output/")

filename = "hp_tune.csv"
df = pd.DataFrame(
    columns=["update", "eta", "alpha", "n_epoch", "bs", "L1", "mean_reg", "var_reg"]
)
df.to_csv("output/" + filename, index=False)

hp = {
    "eta": [1, 2, 5, 10],
    "alpha": [0.5, 1, 2, 3],
    # "n_epoch": [4, 1, 2, 8],
    # "bs": [1, 4, 8],
    # "L1": [1, 0.5],
}

# Init problem
pb = PCconfig(mini=False)
sparse = True
relax = False
n_users = 10
debug = False

# Define fixed training parameters
n_steps = 100
lbda_indif = 1  # math.log(10)/utility_diff #pba of indif<0.1 if |u1-u2| > utility_diff
lbda_pref = lbda_indif
val_set = [pb.random_solution() for _ in range(10)]
regret_step = 5
QS = "ensemble"
n_model = 25 if QS == "ensemble" else 1
diverse = False


for param in hp.keys():
    print(param)
    for value in hp[param]:
        eta = value if param == "eta" else 2
        alpha = value if param == "alpha" else 0.5
        bs = value if param == "bs" else 4
        n_epoch = value if param == "n_epoch" else 4
        L1 = value if param == "L1" else 0.5

        if param == "L1":
            weight_update = ["MLE_L1"]
        elif (param == "n_epoch") or (param == "bs"):
            weight_update = ["MLE", "MLE_L1"]
        else:
            weight_update = ["SP", "NCE", "MLE"]  # , "MLE_L1"]

        # Metrics
        total_train_time = 0
        total_metrics = dict()
        for update in weight_update:
            total_metrics[update] = {"val": [], "regret": [], "cs": [], "n_queries": []}

        for seed in range(100, 100 + n_users):
            # randomly defines true weights
            user = User(
                seed, pb=pb, sparse=sparse, lbda_pref=lbda_pref, lbda_indif=lbda_indif
            )
            best_utility, _ = user.preferred_sol()

            w, metrics = active_learning_loop(
                pb=pb,
                eta=eta,
                n_steps=n_steps,
                user=user,
                best_utility=best_utility,
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
            )

            for update in weight_update:
                for m in metrics[update].keys():
                    total_metrics[update][m].append(np.array(metrics[update][m]))

        for update in metrics.keys():
            mu, se, cumul_reg, n_queries, user_sat, n_query_for_sat = compute_metrics(
                total_metrics, update
            )

            df = pd.read_csv("output/" + filename)
            new_row = pd.DataFrame(
                {
                    "update": update,
                    "eta": eta,
                    "alpha": alpha,
                    "n_epoch": n_epoch,
                    "bs": bs,
                    "L1": L1,
                    "mean_reg": mu,
                    "n_queries": n_queries,
                    "user_sat": user_sat,
                },
                index=[0],
            )
            df = pd.concat((df, new_row), ignore_index=True)
            df.to_csv("output/" + filename, index=False)
