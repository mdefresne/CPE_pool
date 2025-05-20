import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum, Model
import pandas as pd
import numpy as np


################### Adapted from https://github.com/BastianVT/EfficientSP ######################
def solvePCTSP(X, c, norm=np.ones(5), QSinfo=None):

    (
        N,
        V,
        K,
        Q,
        M,
        q,
        Prize_collected,
        Cost_Penalty,
        True_Weight_Matrix,
        Minimum_Prize,
        Ma,
        SubObj,
        num,
        c0,
    ) = X

    cW = c[0:4]
    pW = c[4]

    c3 = [[0 for j in range(0, len(V))] for i in range(0, len(V))]
    c3_no_w = [
        [[] for j in range(0, len(V))] for i in range(0, len(V))
    ]  # sobj without weights
    for i in range(0, len(cW)):
        for j in range(0, len(V)):
            for k in range(0, len(V)):
                c3[j][k] = c3[j][k] + cW[i] * SubObj[i].loc[j].iloc[k] * norm[i]
                c3_no_w[j][k].append(SubObj[i].loc[j].iloc[k] * norm[i])
    c3 = list(np.array(c3).reshape((len(V) * len(V))))
    c3_no_w = list(np.array(c3_no_w).reshape((len(V) * len(V), len(cW))))
    cdict = dict(
        zip([(i, j) for i in range(0, int(len(V))) for j in range(0, int(len(V)))], c3)
    )
    cdict_no_w = dict(
        zip(
            [(i, j) for i in range(0, int(len(V))) for j in range(0, int(len(V)))],
            c3_no_w,
        )
    )

    if QSinfo is not None:
        selected_sol, y1, phi1, gamma = QSinfo
    else:
        selected_sol, y1 = None, None

    # with gp.Env(empty=True) as env:
    with gp.Env(params=get_licence()) as env:
        env.setParam("OutputFlag", 0)
        env.start()

        with gp.Model("VRP", env=env) as mdl:

            """Decision Variable"""
            # Arcs
            x = mdl.addVars([(i, j) for i in V for j in V], vtype=GRB.BINARY, name="x")

            """Constraints"""
            # Flow
            mdl.addConstrs(
                quicksum(x[i, j] for i in V) - quicksum(x[j, i] for i in V) == 0
                for j in V
            )

            for i in V:
                mdl.addConstr(quicksum(x[i, j] for j in V) <= 1)
                mdl.addConstr(quicksum(x[j, i] for j in V) <= 1)

            # start from warehouse
            mdl.addConstr(quicksum(x[0, j] for j in V) == 1)

            # Prize collected
            mdl.addConstr(
                quicksum(x[i, j] * Prize_collected[j] for i in V for j in V)
                >= Minimum_Prize
            )

            # you dont go to yourself"
            mdl.addConstrs(x[i, i] == 0 for i in V)

            u = mdl.addVars([i for i in V], name="u")
            mdl.addConstrs(
                u[j] - u[i] >= q[i] - Q[1] * (1 - x[i, j]) for i in N for j in N
            )
            mdl.addConstrs(q[i] <= u[i] for i in N)
            mdl.addConstrs(u[i] <= Q[1] for i in N)

            # auxiliary var indicating if a city has been visited
            s = mdl.addVars([i for i in V], vtype=GRB.BINARY, name="s")
            # mdl.addConstrs([i in 1:n], sum(x[i, j] for j in setdiff(1:n, [i])) - s[i] == 0)
            mdl.addConstrs(quicksum(x[i, j] for j in V) - s[i] == 0 for i in V)
            mdl.addConstrs(quicksum(x[i, j] for i in V) - s[j] == 0 for j in V)

            """'Objective Function"""
            if y1 is None:
                mdl.setObjective(
                    quicksum(x[i, j] * cdict[(i, j)] for i in V for j in V)
                    + norm[-1]
                    * quicksum(
                        # pW*(1-x[i, j])*Cost_Penalty[j] for i in N for j in N)
                        pW * (1 - s[i]) * Cost_Penalty[i]
                        for i in V
                    ),
                    GRB.MINIMIZE,
                )

            else:  # choice perceptron objective
                # y2 != y1
                y1 = y1.reshape(len(V), len(V))
                # auxiliary variable to count the number of different values
                count = mdl.addVars(
                    [(i, j) for i in V for j in V], vtype=GRB.BINARY, name="c"
                )
                # auxiliary var equals to y
                y = mdl.addVars(
                    [(i, j) for i in V for j in V], vtype=GRB.BINARY, name="y"
                )
                mdl.addConstrs(y[i, j] == y1[i, j] for i in V for j in V)
                mdl.addConstrs(
                    count[i, j] == gp.and_(x[i, j], y[i, j]) for i in V for j in V
                )
                mdl.addConstr(
                    quicksum(count[i, j] for i in V for j in V) <= np.sum(y1) - 1
                )

                phi2 = quicksum(
                    x[i, j] * cdict_no_w[(i, j)] for i in V for j in V
                )  # len 4
                z = mdl.addVars([i for i in range(5)], name="z")
                p = mdl.addVars([i for i in range(5)], name="p")
                for i in range(5):
                    mdl.addConstr(z[i] == gp.abs_(p[i]))  # need 2 variables
                    if i == 4:
                        phi = norm[-1] * quicksum(
                            (1 - s[i]) * Cost_Penalty[i] for i in V
                        )
                    else:
                        phi = phi2[i]
                    mdl.addConstr(phi - phi1[i] == p[i])
                delta = z.sum()

                mu = quicksum(x[i, j] * cdict[(i, j)] for i in V for j in V)
                mu += norm[-1] * quicksum(pW * (1 - s[i]) * Cost_Penalty[i] for i in V)

                mdl.setObjective(gamma * delta + (1 - gamma) * (-mu), GRB.MAXIMIZE)

            """Solve"""
            mdl.Params.MIPGap = 0
            mdl.Params.TimeLimit = 300  # seconds

            mdl.optimize()

            # retrieve solution
            # print(f"Obj: {mdl.ObjVal:g}")
            vals = mdl.getAttr("x", x)
            selected = [
                gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)[i]
                for i in range(round(sum(vals.values())))
            ]

            return selected


def get_licence(file_path="../WLS_licence/gurobi.lic"):
    f = open(file_path, "r")
    L = f.readlines()
    f.close()

    options = {
        "WLSACCESSID": L[3].strip().split("=")[-1],
        "WLSSECRET": L[4].strip().split("=")[-1],
        "LICENSEID": int(L[5].strip().split("=")[-1]),
    }

    return options


def createPCTSPdata(text):

    content = text

    n = len(content.split("\n")[0].split())
    L2 = len(content.split("\n")[6].split())

    Prize_collected = [0] + [int(content.split("\n")[0].split()[i]) for i in range(n)]

    Weight_Penalty = int(content.split("\n")[2].split()[0])

    Cost_Penalty = [0] + [int(content.split("\n")[4].split()[i]) for i in range(n)]

    True_Weight_Matrix = [float(content.split("\n")[6].split()[i]) for i in range(L2)]

    True_Weight_Matrix.append(Weight_Penalty)

    Minimum_Prize = int(content.split("\n")[8].split()[0])

    def FormatMatrix(MatrixA, n):
        MatrixA = MatrixA.rename(
            index=dict(enumerate(range(1, n + 1))),
            columns=dict(enumerate(range(1, n + 1))),
        )
        MatrixA.loc["0"] = 0
        MatrixA["0"] = 0
        MatrixA = MatrixA[list(MatrixA.columns[-1:]) + list(MatrixA.columns[:-1])]
        MatrixA = MatrixA.reindex(range(n + 1))
        MatrixA.loc[0] = 0
        return MatrixA

    def read_matrix(content, n, b_inf, b_sup):
        MatrixA = pd.DataFrame(
            pd.DataFrame(
                [
                    int(content.split("\n")[b_inf:b_sup][j].split("\t")[i])
                    for j in range(n)
                    for i in range(n)
                ]
            ).values.reshape(n, n)
        )
        return FormatMatrix(MatrixA, n)

    MatrixA = read_matrix(content, n, b_inf=10, b_sup=10 + n)
    MatrixB = read_matrix(content, n, b_inf=10 + n + 1, b_sup=10 + 2 * n + 1)
    MatrixC = read_matrix(content, n, b_inf=10 + 2 * n + 2, b_sup=10 + 3 * n + 2)
    MatrixD = read_matrix(content, n, b_inf=10 + 3 * n + 3, b_sup=10 + 4 * n + 3)

    SubObj = [MatrixA, MatrixB, MatrixC, MatrixD]

    N = [i for i in range(0, n)]
    # Clients and warehouse
    V = N + [n]
    # vehicle type
    K = [1]
    # capacities
    Q = {1: 1000}
    # number of vehicles
    M = {1: 1}
    # demand
    q = {i: np.random.randint(1, 2) for i in range(0, n + 1)}

    X = [
        N,
        V,
        K,
        Q,
        M,
        q,
        Prize_collected,
        Cost_Penalty,
        True_Weight_Matrix,
        Minimum_Prize,
        MatrixA,
        SubObj,
        0,
        0,
    ]

    return X  # , solvePCTSP(X, True_Weight_Matrix)


###################################################################################################


class PC_TSP(object):
    def __init__(self, size, n_obj):
        """n is the size of the instances"""

        self.n_obj = n_obj
        self.size = size
        # Fill in necessary attribute with nul values
        self.X = None
        self.N = np.arange(size)
        self.V = np.arange(size + 1)
        self.Cost_Penalty = np.zeros(size + 1)
        self.SubObj = [np.zeros((size + 1, size + 1)) for _ in range(self.n_obj)]
        self.edge_costs = np.zeros((size + 1, size + 1, self.n_obj - 1))
        self.norm = 1
        self.name = "TSP_" + str(self.size)

    def features(self, y):

        n = self.n_obj - 1
        x = np.array(y).reshape(len(self.V), len(self.V))
        s = np.any(x, axis=0)
        x = np.repeat(x, n).reshape(len(self.V), len(self.V), n)

        # - to maximize utility (and keep minimizing the objective)
        obj = -np.sum(np.multiply(x, self.edge_costs), axis=(0, 1))
        collected_price = -np.dot((1 - s), self.Cost_Penalty)
        obj = np.concatenate((obj, np.array(collected_price).reshape(1)))
        obj = np.multiply(np.array(obj), self.norm)

        return obj * 10

    def objective(self, y, weights):

        obj = np.dot(self.features(y), weights)

        return obj

    def random_solution(self):
        """Generate cycle (without sub-cycles)"""
        seq = [0]
        while seq[0] == 0:
            seq = np.random.choice(self.V, size=len(self.V), replace=False)

        y, start, i = [], 0, 0
        end = seq[0]
        while end != 0:
            end = seq[i]
            y.append((start, end))
            start = end
            i += 1

        return self.to_long_rep(y)

    def to_long_rep(self, y):

        long_sol = np.zeros((len(self.V), len(self.V)))
        for i, j in y:
            long_sol[i, j] = 1

        return long_sol.flatten()

    def to_short_rep(self, y1):

        y = y1.reshape(len(self.V), len(self.V))
        start, end = 0, -1
        L = []
        while end != 0:
            end = np.argmax(y[start])
            L.append((start, end))
            start = end

        return L

    def solve(self, weights, relax=False, QSinfo=None):
        pred_sol = solvePCTSP(self.X, weights, self.norm, QSinfo)
        pred_sol = self.to_long_rep(pred_sol)
        best_utility = self.objective(pred_sol, weights)

        return pred_sol, best_utility


class PC_TSP_instance(PC_TSP):
    def __init__(self, file, PC_TSP_pb):

        super().__init__(size=PC_TSP_pb.size, n_obj=PC_TSP_pb.n_obj)
        file_path = f"data/pctsp/size_{PC_TSP_pb.size}/" + file
        with open(file_path, "r") as f:
            text = f.read()
            self.X = createPCTSPdata(text)

        (
            self.N,
            self.V,
            _,
            _,
            _,
            _,
            _,
            self.Cost_Penalty,
            self.True_Weight_Matrix,
            _,
            _,
            self.SubObj,
            _,
            _,
        ) = self.X

        assert self.n_obj == len(self.SubObj) + 1
        self.edge_costs = self._edge_costs()
        self.norm = self._normalize()
        self.name = "TSP_" + str(len(self.N))

    def _edge_costs(self):
        """Translate edge cost data from pandas in SubObj to numpy array of shape V*V*N_obj."""
        assert self.n_obj - 1 == len(self.SubObj)
        edge_costs = [np.array(self.SubObj[i]) for i in range(self.n_obj - 1)]
        edge_costs = np.swapaxes(np.array(edge_costs), 0, 2)
        edge_costs = np.transpose(edge_costs, (1, 0, 2))

        return edge_costs

    def _normalize(self):
        """Compute normalization coefficients"""

        # norm = np.sum(self.edge_costs, axis=(0, 1))
        # norm = list(norm) + [np.sum(self.Cost_Penalty)]
        norm = [
            np.sum(np.max(self.edge_costs[:, :, i], axis=1))
            for i in range(self.n_obj - 1)
        ]
        norm += [np.sum(self.Cost_Penalty)]

        return 1 / np.array(norm)
