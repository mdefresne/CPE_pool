import numpy as np
import cpmpy as cp
import json


class CPproblem(object):
    """
    Define a CP problem given cpmpy variables and constraints.
    This class should be inherited by any problem example.
    """

    def __init__(self, variables, constraints, name="CPproblem"):
        self.variables = variables
        self.constraints = constraints
        self.name = name

        self.model = cp.Model()
        for cst in constraints:
            self.model += cst
        self.relaxed_model = cp.Model(cp.Count(self.variables, 0) >= 0)

    def get_full_model(self):
        return self.variables, self.model

    def sub_objectives(self, y):
        """
        To be overriden.
        Define sub-objectives for the solution y (can be a np.array or cpmpy variable).
        Must return: - list of sub-objectives (list of cpmpy constraints)
                     - Normalization weights (np.array of same length)
        """

        return [], []

    def objective(self, y, weight):
        """
        Define the complete objective function.
        Input a cpmpy variables or a solution (np.array) and a nunpy array of weights.
        """
        subobj, normalize = self.sub_objectives(y)
        norm_weight = np.multiply(normalize, weight)
        obj = sum([subobj[i] * norm_weight[i] for i in range(self.n_obj)])

        return obj

    def features(self, y):
        sobj, normalize = self.sub_objectives(y)
        phi = np.multiply(sobj, normalize)

        return phi

    def solve(self, weights, relax=False):
        """Compute best choices given weight."""
        model = self.relaxed_model if relax else self.model
        model.maximize(self.objective(self.variables, weights.squeeze()))
        has_sol = model.solve()  # ("gurobi", time_limit=100)
        if has_sol:
            pred_sol = self.variables.value()
        else:
            print("No solution found")
            return 1, None

        best_utility = model.objective_value()
        return pred_sol, best_utility


class Roster(CPproblem):
    def __init__(self):
        # Data
        self.n_days = 7
        self.n_doctors = 5
        self.n_shifts = 4
        self.Appt, self.Call, self.Oper, self.Free = range(self.n_shifts)

        # model
        roster = cp.intvar(0, self.n_shifts - 1, shape=(self.n_doctors, self.n_days))
        constraints = [
            sum(roster == self.Call) >= 1,  # at least one call per day
            sum(roster[:, : self.n_days - 2] == self.Oper)
            <= 2,  # less than 2 operations per weekday
            cp.Count(roster, self.Oper) >= 7,  # at least 7 operations per week
            cp.Count(roster, self.Appt) >= 4,  # at least 4 appointments per week
        ]
        constraints += [
            ((roster[:, j] == self.Oper).implies(roster[:, j + 1] == self.Free))
            for j in range(self.n_days - 1)
        ]

        name = "Roster"
        super().__init__(roster, constraints, name)

        self.n_obj = self.sub_objectives(roster, return_num=True)

    def sub_objectives(self, y, return_num=False):
        """
        Define sub-objectives for the solution y (can be a np.array or cpmpy variable):
        4 such that each type of task is farily allocated and 4 counting tasks on the weekend
        Output the list of sub_objectives and a numpy array of normalization terms
        """
        fair_tasks = [
            (
                cp.Maximum([cp.Count(y[p, :], task) for p in range(self.n_doctors)])
                - cp.Minimum([cp.Count(y[p, :], task) for p in range(self.n_doctors)])
            )
            for task in (self.Appt, self.Call, self.Oper, self.Free)
        ]
        normalize = [1 / (self.n_days)] * len(fair_tasks)

        we_tasks = [
            sum([cp.Count(y[p, 5:], task) for p in range(self.n_doctors)])
            for task in (self.Appt, self.Call, self.Oper, self.Free)
        ]
        normalize += [1 / (self.n_doctors * 2)] * len(we_tasks)

        sobj = fair_tasks + we_tasks
        if return_num:
            return len(sobj)

        if not isinstance(y, cp.expressions.variables.NDVarArray):
            # return the value of each subojectives for the specific solution y
            sobj = np.array([s.value() for s in sobj])

        return sobj, normalize

    def random_solution(self):

        return np.random.randint(self.n_shifts, size=(self.n_doctors, self.n_days))

    def pretty_print(self):
        """By Tias Guns"""
        shift_map = {self.Appt: "A", self.Call: "C", self.Oper: "O", self.Free: "F"}
        mapped = np.array(
            [[shift_map[val] for val in row] for row in self.roster.value()]
        )
        print(mapped)
        return


class PCconfig(CPproblem):
    """Data from https://github.com/stefanoteso/setmargin"""

    def __init__(self, mini=False, idx=0, int_cost=False):

        datafile = "data/PCconfig_mini.json" if mini else "data/PCconfig.json"
        with open(datafile, "r") as f:
            data = json.load(f)
        self.domain_of = data["domain_of"]
        self.costs_of = data["costs_of"]

        assert len(self.domain_of) == len(self.costs_of)
        for key in self.domain_of.keys():
            assert len(self.domain_of[key]) == len(self.costs_of[key])

        self.variables = cp.cpm_array(
            [
                cp.intvar(
                    0, len(self.domain_of[compound]) - 1, name=compound + str(idx)
                )
                for compound in self.domain_of.keys()
            ]
        )
        self.compound_map = dict()
        for i, key in enumerate(self.domain_of.keys()):
            self.compound_map[key] = i

        constraints = []
        for cst_type in data["constraints"]:
            for cst in data["constraints"][cst_type]:
                var1, var2 = cst
                constraints.append(self.to_constraints(var1, var2))

        name = "PCmini" if mini else "PC"
        super().__init__(self.variables, constraints, name)

        self.n_obj = self.sub_objectives(self.variables, return_num=True)

        if int_cost:
            for compound in self.costs_of.keys():
                self.costs_of[compound] = [int(c) for c in self.costs_of[compound]]

    def to_constraints(self, var1, var2):
        """
        Values of var1 implies values of var 2
        var1/var2 is a tuple (str, list) where str is the name of the compound and list the possible values.
        example: ("Type", ["Laptop", "Desktop"])
        """
        compound1, value1 = var1
        compound2, value2 = var2
        domain1 = [self.domain_of[compound1].index(v) for v in value1]
        domain2 = [self.domain_of[compound2].index(v) for v in value2]
        compound1_idx = self.compound_map[compound1]
        compound2_idx = self.compound_map[compound2]
        cst = cp.InDomain(self.variables[compound1_idx], domain1).implies(
            cp.InDomain(self.variables[compound2_idx], domain2)
        )

        return cst

    def sub_objectives(self, y, return_num=False):
        """
        Define sub-objectives for the solution y (dict of list or dict of CPMpy variables):
        Price + individual preference over each value of each attribute
        """

        sobj = []
        for compound in self.domain_of.keys():
            for value in self.domain_of[compound]:
                compound_idx = self.compound_map[compound]
                sobj.append(y[compound_idx] == self.domain_of[compound].index(value))
        normalize = [1] * len(sobj)

        costs = []
        for compound in self.costs_of.keys():
            costs += self.costs_of[compound]
        price = sum(np.array(costs) * np.array(sobj))
        max_price = sum(
            [max(self.costs_of[compound]) for compound in self.costs_of.keys()]
        )

        sobj.append(price)
        normalize.append(1 / max_price)

        if return_num:
            return len(sobj)
        return np.array(sobj), normalize

    def random_solution(self):

        y = np.zeros(len(self.variables))
        for i in range(len(y)):
            y[i] = np.random.randint(self.variables[i].ub + 1)

        return y


class PCconfig_MO(PCconfig):
    """Multi-objective variant of the PC configuration problem."""

    def __init__(self, mini=False):

        super().__init__(mini)
        self.n_obj = self.sub_objectives(self.variables, return_num=True)

    def sub_objectives(self, y, return_num=False):
        """
        Define sub-objectives for the solution y (dict of list or dict of CPMpy variables):
        Categorical features (manufacturer and type) are unchanged
        Other features are considered as value (eg, freauency of CPU rather than its name
        """

        cat_ft = ["Manufacturer", "Type"]
        L_sobj, normalize = [], []
        price = 0

        for compound in self.domain_of.keys():
            domain = self.domain_of[compound]
            compound_idx = self.compound_map[compound]
            if compound in cat_ft:
                for value in domain:
                    L_sobj.append(y[compound_idx] == domain.index(value))
                    normalize.append(1)
            else:
                if compound == "Memory":
                    domain = list(np.log2(domain))
                if compound == "CPU":
                    domain = [int(value.split("@")[-1]) for value in domain]
                norm = 1 / np.max(domain)
                sobj = sum(
                    [
                        (y[compound_idx] == domain.index(value)) * value
                        for value in domain
                    ]
                )
                L_sobj.append(sobj)
                normalize.append(norm)

            if compound == "CPU":
                domain = self.domain_of[compound]
            price += sum(
                np.array(self.costs_of[compound])
                * np.array([y[compound_idx] == domain.index(value) for value in domain])
            )

        max_price = sum(
            [max(self.costs_of[compound]) for compound in self.costs_of.keys()]
        )
        L_sobj.append(price)
        normalize.append(1 / max_price)

        if return_num:
            return len(L_sobj)
        return L_sobj, normalize


class Fused(CPproblem):

    def __init__(self, pb1, pb2):

        self.pb1 = pb1
        self.pb2 = pb2
        self.variables = pb1.variables, pb2.variables
        self.constraints = pb1.constraints + pb2.constraints
        super().__init__(self.variables, self.constraints, "fused")

    def sub_objectives(self, y):

        n = len(y) // 2
        y1, y2 = y[:n], y[n:]
        obj1, norm1 = self.pb1.sub_objectives(y1)
        obj2, norm2 = self.pb1.sub_objectives(y2)

        return obj1 + obj2, norm1 + norm2
