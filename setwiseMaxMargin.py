# Adapted from https://github.com/stefanoteso/setmargin/

import numpy as np
import itertools as it
import gurobipy as grb
from gurobipy import GRB
import time
import pickle
import json

from sklearn.utils import check_random_state
from sklearn.model_selection import KFold

from user import User
from CPproblem import PCconfig


class Dataset(object):
    """A dataset over the Cartesian product of all attribute domains.

    :param domain_sizes: list of domain sizes.
    :param x_constraints: constraints on the item configurations.
    """

    def __init__(self, domain_sizes, costs, x_constraints):
        self.domain_sizes = domain_sizes
        self.costs = costs
        if costs is not None:
            assert costs.shape == (self.num_reals(), self.num_bools())
        self.x_constraints = x_constraints

    def __str__(self):
        return "Dataset(domain_sizes={} bools={} reals={} constrs={}".format(
            self.domain_sizes, self.num_bools(), self.num_reals(), self.x_constraints
        )

    def num_bools(self):
        return sum(self.domain_sizes)

    def num_reals(self):
        return 0 if self.costs is None else self.costs.shape[0]

    def num_vars(self):
        return self.num_bools() + self.num_reals()

    def get_domain_ranges(self):
        base = 0
        for size in self.domain_sizes:
            yield base, base + size
            base += size

    def get_zs_in_domains(self):
        zs_in_domains, last_z = [], 0
        for domain_size in self.domain_sizes:
            assert domain_size > 1
            zs_in_domains.append(range(last_z, last_z + domain_size))
            last_z += domain_size
        return zs_in_domains

    def is_item_valid(self, x):
        if (x < -1e-10).any():
            return False
        for zs_in_domain in self.get_zs_in_domains():
            if sum(x[zs_in_domain]) != 1:
                return False
        if self.costs is not None:
            x, c = x[: self.num_bools()], x[self.num_bools() :]
            if not (np.dot(self.costs, x) == c).all():
                return False
        if self.x_constraints is not None:
            for head, body in self.x_constraints:
                if x[head] and not any(x[atom] == 1 for atom in body):
                    return False
        return True

    def compose_item(self, x):
        assert x.shape == (self.num_bools(),)
        if self.num_reals() > 0:
            x = np.hstack((x, np.dot(self.costs, x)))
        assert x.shape == (self.num_bools() + self.num_reals(),)
        return x


class SyntheticDataset(Dataset):
    def __init__(self, domain_sizes):
        super(SyntheticDataset, self).__init__(domain_sizes, None, None)


class DebugConstraintDataset(Dataset):
    def __init__(self, domain_sizes, rng=None):
        x_constraints = self._sample_constraints(domain_sizes, check_random_state(rng))
        super(DebugConstraintDataset, self).__init__(domain_sizes, None, x_constraints)

    def _sample_constraints(self, domain_sizes, rng):
        print(sampling, constraints)
        print("--------------------")
        print("domain_sizes =", domain_sizes)
        constraints = []
        for (i, dsi), (j, dsj) in it.product(
            enumerate(domain_sizes), enumerate(domain_sizes)
        ):
            if i >= j:
                continue
            # XXX come up with something smarter
            head = rng.random_integers(0, dsi - 1)
            body = rng.random_integers(0, dsj - 1)
            print("{}:{} -> {}:{}".format(i, head, j, body))
            index_head = sum(domain_sizes[:i]) + head
            index_body = sum(domain_sizes[:j]) + body
            constraints.append((index_head, [index_body]))
        print("constraints =\n", constraints)
        print("--------------------")
        return constraints


class DebugCostDataset(Dataset):
    def __init__(self, domain_sizes, num_costs=2, rng=None):
        rng = check_random_state(rng)
        costs = rng.uniform(0, 1, size=(num_costs, sum(domain_sizes)))
        super(DebugCostDataset, self).__init__(domain_sizes, costs, None)


class PCDataset(Dataset):
    def __init__(self, has_costs=False):

        datafile = "data/PCconfig.json"
        with open(datafile, "r") as f:
            data = json.load(f)
        self.domain_of = data["domain_of"]
        self.costs_of = data["costs_of"]

        self.attributes = self.domain_of.keys()  # sorted(self.domain_of.keys())
        # remove sort to keep same order as in my work
        domain_sizes = [len(self.domain_of[attr]) for attr in self.attributes]

        cost, max_cost = [], 0.0
        assert len(self.domain_of) == len(self.costs_of)
        for attr in self.attributes:
            assert len(self.domain_of[attr]) == len(self.costs_of[attr])
            costs_for_attr = np.array(self.costs_of[attr])
            max_cost += max(costs_for_attr)
            cost.extend(costs_for_attr)
        costs = np.array([cost]) / max_cost

        x_constraints = []
        # Manufacturer->Type
        x_constraints.extend(
            self._to_constraints(
                ("Manufacturer", ["Compaq"]), ("Type", ["Laptop", "Desktop"])
            )
        )
        x_constraints.extend(
            self._to_constraints(("Manufacturer", ["Fujitsu"]), ("Type", ["Laptop"]))
        )
        x_constraints.extend(
            self._to_constraints(("Manufacturer", ["HP"]), ("Type", ["Desktop"]))
        )
        x_constraints.extend(
            self._to_constraints(
                ("Manufacturer", ["Sony"]), ("Type", ["Laptop", "Tower"])
            )
        )

        # Manufacturer->CPU
        x_constraints.extend(
            self._to_constraints(
                ("Manufacturer", ["Apple"]),
                (
                    "CPU",
                    [
                        "PowerPC G3 @266",
                        "PowerPC G3 @300",
                        "PowerPC G3 @400",
                        "PowerPC G3 @450",
                        "PowerPC G3 @500",
                        "PowerPC G3 @550",
                        "PowerPC G3 @600",
                        "PowerPC G3 @700",
                        "PowerPC G4 @700",
                        "PowerPC G4 @733",
                    ],
                ),
            )
        )
        x_constraints.extend(
            self._to_constraints(
                ("Manufacturer", ["Compaq", "Sony"]),
                (
                    "CPU",
                    [
                        "AMD Athlon @1000",
                        "AMD Athlon @1330",
                        "AMD Duron @700",
                        "AMD Duron @900",
                        "Intel Celeron @500",
                        "Intel Celeron @600",
                        "Intel Celeron @800",
                        "Intel Celeron @900",
                        "Intel Celeron @1000",
                        "Intel Celeron @1100",
                        "Intel Celeron @1200",
                        "Intel Celeron @1300",
                        "Intel Celeron @1400",
                        "Intel Celeron @1700",
                        "Intel Pentium @500",
                        "Intel Pentium @600",
                        "Intel Pentium @800",
                        "Intel Pentium @900",
                        "Intel Pentium @1000",
                        "Intel Pentium @1100",
                        "Intel Pentium @1300",
                        "Intel Pentium @1500",
                        "Intel Pentium @1600",
                        "Intel Pentium @1700",
                        "Intel Pentium @1800",
                        "Intel Pentium @2200",
                    ],
                ),
            )
        )
        x_constraints.extend(
            self._to_constraints(
                ("Manufacturer", ["Fujitsu"]),
                (
                    "CPU",
                    [
                        "Crusoe @800",
                        "Intel Celeron @500",
                        "Intel Celeron @600",
                        "Intel Celeron @800",
                        "Intel Celeron @900",
                        "Intel Celeron @1000",
                        "Intel Celeron @1100",
                        "Intel Celeron @1200",
                        "Intel Celeron @1300",
                        "Intel Celeron @1400",
                        "Intel Celeron @1700",
                        "Intel Pentium @500",
                        "Intel Pentium @600",
                        "Intel Pentium @800",
                        "Intel Pentium @900",
                        "Intel Pentium @1000",
                        "Intel Pentium @1100",
                        "Intel Pentium @1300",
                        "Intel Pentium @1500",
                        "Intel Pentium @1600",
                        "Intel Pentium @1700",
                        "Intel Pentium @1800",
                        "Intel Pentium @2200",
                    ],
                ),
            )
        )
        x_constraints.extend(
            self._to_constraints(
                ("Manufacturer", ["Dell", "Gateway", "Toshiba"]),
                (
                    "CPU",
                    [
                        "Intel Celeron @500",
                        "Intel Celeron @600",
                        "Intel Celeron @800",
                        "Intel Celeron @900",
                        "Intel Celeron @1000",
                        "Intel Celeron @1100",
                        "Intel Celeron @1200",
                        "Intel Celeron @1300",
                        "Intel Celeron @1400",
                        "Intel Celeron @1700",
                        "Intel Pentium @500",
                        "Intel Pentium @600",
                        "Intel Pentium @800",
                        "Intel Pentium @900",
                        "Intel Pentium @1000",
                        "Intel Pentium @1100",
                        "Intel Pentium @1300",
                        "Intel Pentium @1500",
                        "Intel Pentium @1600",
                        "Intel Pentium @1700",
                        "Intel Pentium @1800",
                        "Intel Pentium @2200",
                    ],
                ),
            )
        )
        x_constraints.extend(
            self._to_constraints(
                ("Manufacturer", ["HP"]),
                (
                    "CPU",
                    [
                        "Intel Pentium @500",
                        "Intel Pentium @600",
                        "Intel Pentium @800",
                        "Intel Pentium @900",
                        "Intel Pentium @1000",
                        "Intel Pentium @1100",
                        "Intel Pentium @1300",
                        "Intel Pentium @1500",
                        "Intel Pentium @1600",
                        "Intel Pentium @1700",
                        "Intel Pentium @1800",
                        "Intel Pentium @2200",
                    ],
                ),
            )
        )

        # Type->Memory
        x_constraints.extend(
            self._to_constraints(
                ("Type", ["Laptop"]),
                ("Memory", [64, 128, 160, 192, 256, 320, 384, 512, 1024]),
            )
        )
        x_constraints.extend(
            self._to_constraints(
                ("Type", ["Desktop"]), ("Memory", [128, 256, 512, 1024])
            )
        )
        x_constraints.extend(
            self._to_constraints(
                ("Type", ["Tower"]), ("Memory", [256, 512, 1024, 2048])
            )
        )

        # Type->HDSize
        x_constraints.extend(
            self._to_constraints(
                ("Type", ["Desktop", "Tower"]), ("HDSize", [20, 30, 40, 60, 80, 120])
            )
        )
        x_constraints.extend(
            self._to_constraints(
                ("Type", ["Laptop"]), ("HDSize", [8, 10, 12, 15, 20, 30])
            )
        )

        # Type->Monitor
        x_constraints.extend(
            self._to_constraints(
                ("Type", ["Desktop", "Tower"]), ("Monitor", [15, 17, 21])
            )
        )
        x_constraints.extend(
            self._to_constraints(
                ("Type", ["Laptop"]), ("Monitor", [10, 10.4, 12, 13.3, 14, 15])
            )
        )

        super(PCDataset, self).__init__(
            domain_sizes, costs if has_costs else None, x_constraints
        )

    # error in the code: ('Manufacturer', 'Apple') and ('Type', 'Desktop') both are 16
    def _attr_value_to_bit(self, attr, value):
        base, i = 0, None
        for attr_j in self.attributes:
            if attr == attr_j:
                assert value in self.domain_of[attr]
                i = self.domain_of[attr].index(value)
                break
            # base += len(self.domain_of[attr]) #ERROR!!
            base += len(self.domain_of[attr_j])
        assert i is not None
        return base + i

    def _to_constraints(self, body_vars, head_vars):
        constraints = []
        body_attr, body_vals = body_vars
        head_attr, head_vals = head_vars
        for body_val in body_vals:
            constraints.append(
                (
                    self._attr_value_to_bit(body_attr, body_val),
                    [
                        self._attr_value_to_bit(head_attr, head_val)
                        for head_val in head_vals
                    ],
                )
            )
        return constraints

def add_item_constraints(model, dataset, x):
    """Add one-hot and Horn item constraints.

    :param model: the Gurobi model.
    :param dataset: the dataset.
    :param x: a list of Guorbi expressions.
    """
    for zs_in_domain in dataset.get_zs_in_domains():
        model.addConstr(grb.quicksum([x[z] for z in zs_in_domain]) == 1)

    if dataset.x_constraints is not None:
        for body, head in dataset.x_constraints:
            model.addConstr(
                (1 - x[body]) + grb.quicksum([x[atom] for atom in head]) >= 1
            )


def compute_setmargin(dataset, answers, set_size, alphas, debug=True):
    """Adapted from https://github.com/stefanoteso/setmargin/"""
    num_bools = dataset.num_bools()
    num_reals = dataset.num_reals()
    num_examples = len(answers)  # k

    w_top = np.ones(num_bools)
    if num_reals > 0:
        w_top += np.dot(dataset.costs.T, np.ones(num_reals))
    w_max = np.max(w_top)

    model = grb.Model("setmargin")
    model.params.Seed = 0
    # if self._threads is not None:
    #   model.params.Threads = self._threads
    model.params.OutputFlag = 0

    # Declare the variables
    ws, xs = {}, {}
    for i in range(set_size):
        for z in range(num_bools):
            ws[i, z] = model.addVar(vtype=GRB.CONTINUOUS, name="w_{}_{}".format(i, z))
            xs[i, z] = model.addVar(vtype=GRB.BINARY, name="x_{}_{}".format(i, z))

    slacks = {}
    for i in range(set_size):
        for k in range(num_examples):
            slacks[i, k] = model.addVar(
                vtype=GRB.CONTINUOUS, name="slack_{}_{}".format(i, k)
            )

    ps = {}
    for i in range(set_size):
        for j in range(set_size):
            for z in range(num_bools):
                ps[i, j, z] = model.addVar(
                    vtype=GRB.CONTINUOUS, name="p_{}_{}_{}".format(i, j, z)
                )

    multimargin = False
    if not multimargin:
        margins = [model.addVar(vtype=GRB.CONTINUOUS, name="margin")]
    else:
        margins = [
            model.addVar(vtype=GRB.CONTINUOUS, name="margin_on_ys"),
            model.addVar(vtype=GRB.CONTINUOUS, name="margin_on_xs"),
        ]

    model.modelSense = GRB.MAXIMIZE
    model.update()

    # Define the objective function
    if False:
        # XXX getting temp[0] below 1 makes the set_size=1 problem
        # become unbounded; avoid normalizing the hyperparameters
        # for now
        temp = (
            0.0 if len(slacks) == 0 else alphas[0] / (set_size * num_examples),
            alphas[1] / set_size,
            alphas[2] / set_size,
        )
    else:
        temp = alphas

    obj_margins = grb.quicksum(margins)

    obj_slacks = 0
    if len(slacks) > 0:
        obj_slacks = temp[0] * grb.quicksum(slacks.values())

    obj_weights = temp[1] * grb.quicksum(ws.values())

    obj_scores = temp[2] * grb.quicksum(
        [ps[i, i, z] for i in range(set_size) for z in range(num_bools)]
    )

    model.setObjective(obj_margins - obj_slacks - obj_weights + obj_scores)

    # Add the various constraints
    # Eq. 9
    for i in range(set_size):
        for k in range(num_examples):
            x1, x2, ans = answers[k]
            assert x1.shape == (num_bools,)
            assert x2.shape == (num_bools,)
            assert ans in (-1, 0, 1)

            diff = x1 - x2 if ans >= 0 else x2 - x1
            dot = grb.quicksum([ws[i, z] * diff[z] for z in range(num_bools)])

            if ans == 0:
                # Only one of dot and -dot is positive, and the slacks are
                # always positive, so this should work fine as a replacement
                # for abs(dot) <= slacks[i,j]
                model.addConstr(dot <= slacks[i, k])
                model.addConstr(-dot <= slacks[i, k])
            else:
                model.addConstr(dot >= (margins[0] - slacks[i, k]))

    # Eq. 10
    for i in range(set_size):
        # for j in range(i) + range(i+1, set_size):
        for j in [_ for _ in range(i)] + [_ for _ in range(i + 1, set_size)]:
            score_diff = grb.quicksum(
                [ps[i, i, z] - ps[i, j, z] for z in range(num_bools)]
            )
            model.addConstr(score_diff >= margins[-1])

    # Eq. 11
    for i in range(set_size):
        for z in range(num_bools):
            model.addConstr(ps[i, i, z] <= (w_max * xs[i, z]))

    # Eq. 12
    for i in range(set_size):
        for z in range(num_bools):
            model.addConstr(ps[i, i, z] <= ws[i, z])

    # Eq. 13
    for i in range(set_size):
        for j in [_ for _ in range(i)] + [_ for _ in range(i + 1, set_size)]:
            for z in range(num_bools):
                model.addConstr(ps[i, j, z] >= (ws[i, z] - 2 * w_max * (1 - xs[j, z])))

    # Eq. 15
    for i in range(set_size):
        for z in range(num_bools):
            model.addConstr(ws[i, z] <= w_max)

    # Eq. 18a
    for i in range(set_size):
        for j in range(set_size):
            for z in range(num_bools):
                model.addConstr(ps[i, j, z] >= 0)

    # Eq. 18b
    for i in range(set_size):
        for z in range(num_bools):
            model.addConstr(ws[i, z] >= 0)

    # Eq. 19
    for i in range(set_size):
        for k in range(num_examples):
            model.addConstr(slacks[i, k] >= 0)

    # Eq. 20
    for margin in margins:
        model.addConstr(margin >= 0)
        if set_size == 1 and all(ans == 0 for _, _, ans in answers):
            # XXX work around the fact that if we only have one hyperplane and
            # the user is indifferent to everything we throwed at her, the margin
            # will not appear in any constraint and thus the problem will be
            # unbounded.
            model.addConstr(margin == 0)
    if multimargin:
        if all(ans == 0 for _, _, ans in answers):
            model.addConstr(margins[0] == 0)
        if set_size == 1:
            model.addConstr(margins[1] == 0)

    for i in range(set_size):
        x = [xs[(i, z)] for z in range(num_bools)]
        add_item_constraints(model, dataset, x)

    model.update()
    # self._dump_model(model, "setmargin_full")

    try:
        model.optimize()
        _ = model.objVal
    except:
        print(
            """\
            optimization failed!
    
            answers =
            {}
    
            set_size = {}
            status = {}
            alphas = {}
            """
        ).format(answers, set_size, STATUS_TO_REASON[model.status], temp)
        # raise RuntimeError(message)

    output_ws = np.zeros((set_size, num_bools))
    output_xs = np.zeros((set_size, num_bools))
    for i in range(set_size):
        for z in range(num_bools):
            output_ws[i, z] = ws[i, z].x
            output_xs[i, z] = xs[i, z].x

    for i in range(set_size):
        x = np.array([xs[i, z].x for z in range(num_bools)])
        # assert dataset.is_item_valid(dataset.compose_item(x))

    output_ps = np.zeros((set_size, set_size, num_bools))
    output_scores = np.zeros((set_size, set_size))
    for i in range(set_size):
        for j in range(set_size):
            for z in range(num_bools):
                output_ps[i, j, z] = ps[i, j, z].x
                output_scores[i, j] += ps[i, j, z].x

    if len(answers):
        output_slacks = np.zeros((set_size, len(answers)))
    else:
        output_slacks = []
    for i in range(set_size):
        for k in range(num_examples):
            output_slacks[i, k] = slacks[i, k].x

    output_margins = [margin.x for margin in margins]

    if any(np.linalg.norm(w) == 0 for w in output_ws) and debug:
        print("Warning: null weight vector found!")

    debug_scores = np.dot(output_ws, output_xs.T)
    if (np.abs(output_scores - debug_scores) >= 1e-5).any() and debug:
        print("Warning: solver and debug scores mismatch: ")
        print(f"scores = {output_scores} debug scores = {debug_scores}")

    return output_ws, output_xs


def crossvalidate(dataset, answers, set_size, debug=True):
    loss_alphas = []
    for alphas in ALL_ALPHAS:
        kf = KFold(n_splits=NUM_FOLDS, random_state=None, shuffle=False)
        # kfold = KFold(len(answers), n_splits=NUM_FOLDS)
        losses = []
        for train_set, test_set in kf.split(np.arange(len(answers))):
            train_answers = [answers[i] for i in train_set]

            ws, _ = compute_setmargin(dataset, train_answers, set_size, alphas, debug)

            test_answers = [answers[i] for i in test_set]
            xis = np.array([x for x, _, _ in test_answers])
            xjs = np.array([x for _, x, _ in test_answers])
            ys = np.array([s for _, _, s in test_answers])

            ys_hat = np.sign(np.dot(ws, (xis - xjs).T))
            diff = 0.5 * np.abs(ys - ys_hat)  # the difference is broadcast
            losses.append(diff.sum(axis=1).mean())

        loss_alphas.append((sum(losses) / len(losses), alphas))

    loss_alphas = sorted(loss_alphas)
    alphas = loss_alphas[0][1]
    return alphas


def query(user, xi, xj):

    phi_i = dataset.compose_item(xi)
    phi_j = dataset.compose_item(xj)
    delta_util = user._utility_diff(phi_i, phi_j)

    return user.answer(delta_util)


def query_set(user, xs, old_best_item):
    ranking_mode = "all_pairs"
    num_items, num_features = xs.shape
    answers = [
        (xi, xj, query(user, xi, xj))
        for (i, xi), (j, xj) in it.product(enumerate(xs), enumerate(xs))
        if i < j
    ]
    num_queries = len(answers)

    assert len(answers) > 0
    assert num_queries > 0
    return answers, num_queries


###############################################
##################### Running #################
###############################################

NUM_FOLDS = 5
ALL_ALPHAS = list(
    it.product(
        [20.0, 10.0, 5.0, 1.0],
        [10.0, 1.0, 0.1, 0.01],
        [10.0, 1.0, 0.1, 0.01],
    )
)
alphas = ALL_ALPHAS[0]
answers, info, old_best_item, t = [], [], None, 0
set_size = 2
dataset = PCDataset(has_costs=True)

# Default parameters
max_iterations = 100
max_answers = 100
tol = "auto"  # in experiment, 1e-2
alphas = "auto"
crossval_interval = 5
crossval_set_size = 1
# "tol": 1e-2,
# "threads": cpu_count()
do_crossval = alphas == "auto"
alphas = (1.0, 1.0, 1.0)

pb = PCconfig(mini=False)
sparse = True
relax = False
debug = False
n_users = 20

for seed in range(2, n_users, 1):
    user = User(seed, n_obj=77, sparse=sparse, lbda_pref=1, lbda_indif=1)
    user_w_norm = np.linalg.norm(user.w.ravel())
    best_sol, best_utility = user.preferred_sol(pb)
    best_item = pb.features(best_sol)[:-1]  # remove price
    assert best_item.shape == (dataset.num_bools(),)

    answers, info, old_best_item, t = [], [], None, 0
    n_queries = -1
    L_time=[]
    while True:
        old_time = time.time()

        # Crossvalidate the hyperparameters if required
        if do_crossval and t % crossval_interval == 0 and t >= NUM_FOLDS:
            alphas = crossvalidate(dataset, answers, crossval_set_size, debug=False)

        # Solve the set_size=k case
        t0 = time.time()
        _, xs = compute_setmargin(dataset, answers, set_size, alphas)
        assert xs.shape == (set_size, dataset.num_bools())
        L_time.append(time.time()-t0)

        # Update the user answers
        new_answers, num_queries = query_set(user, xs, old_best_item)
        num_identical_answers = 0
        for xi, xj, sx in new_answers:
            for zi, zj, sz in answers:
                if (xi == zi).all() and (xj == zj).all():
                    num_identical_answers += 1
        if num_identical_answers > 0:
            print(
                "Warning: {} identical (up to sign) answers added!".format(
                    num_identical_answers
                )
            )
        answers.extend(new_answers)
        old_best_item = xs[0] if xs.shape[0] == 1 else None

        elapsed = time.time() - old_time
        # print_answers(user, answers)

        # Solve the set_size=1 case
        _, xs = compute_setmargin(dataset, answers, 1, alphas)
        assert xs.shape == (1, dataset.num_bools())

        # Compute the utility loss
        best_score = np.dot(user.w.ravel(), dataset.compose_item(best_item))
        pred_score = np.dot(user.w.ravel(), dataset.compose_item(xs[0]))
        loss = best_score - pred_score
        info.append((num_queries, loss, elapsed))
        print(t, loss)

        t += 1
        if (t >= max_iterations) or (len(answers) >= max_answers):
            break
        # If the user is satisfied, we are done
        if tol == "auto" and user.answer(loss) == 0:
            # indifference between top and current sol
            n_queries = t
            break
        elif tol != "auto" and loss < tol:
            break

    print('Average Query selection time', np.mean(L_time), np.var(L_time))
    total_metrics = (info, n_queries, best_score)
    with open(f"output/SM/user{seed}.pkl", "wb") as fp:
        pickle.dump(total_metrics, fp, protocol=pickle.HIGHEST_PROTOCOL)
