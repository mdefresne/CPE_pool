import numpy as np
import math


class User(object):
    """Implement a synthetic user."""

    def __init__(
        self,
        seed,
        n_obj,
        sparse=False,
        lbda_pref=1,
        lbda_indif=1,
        problem_name="PC",
    ):

        self.seed = seed
        np.random.seed(self.seed)
        self.n_obj = n_obj
        if "TSP" in problem_name:
            self.w = (np.random.dirichlet(np.ones(4), size=1) * 100).tolist()[0]
            self.w += [np.random.randint(1, 100 + 1)]
            self.w = np.array(self.w)
        else:
            self.mu = 25  # np.random.randint(10, 100)
            self.w = np.round(np.random.normal(self.mu, self.mu / 3, self.n_obj), 2)

        self.lbda_pref = lbda_pref
        self.lbda_indif = lbda_indif

        if isinstance(sparse, float):  # (put sparse% of weights to 0)
            choices = np.random.choice(
                np.arange(self.n_obj - 1), int(sparse * (self.n_obj)), replace=False
            )
            to_0 = np.zeros(self.n_obj, dtype="bool")
            to_0[choices] = True
            self.w[to_0] = 0

    def preferred_sol(self, pb):
        """Compute preferred object and its utility."""

        return pb.solve(self.w, relax=False)

    def _utility_diff(self, phi1, phi2):

        true_util1, true_util2 = np.dot(self.w, phi1), np.dot(self.w, phi2)
        return true_util1 - true_util2

    def answer(self, delta_util, val=False):
        """Returns 0 (indifference), 1 (phi1 is preferred) or -1 (phi2 is preferred)."""

        pba_pref, pba_indif = self._BT_pba(delta_util)
        if val:
            if pba_indif > 0.5:
                return 0

        if np.random.random() < pba_indif:  # indifference
            return 0
        else:
            if np.random.random() < pba_pref:
                return 1
        return -1

    def _BT_pba(self, delta_util):
        """
        Given the utility of two solutions,
        return the proba the solution 1 is prefered and the proba of indifference
        based on Bradley-Terry model.
        """
        pba_pref = 1 / (1 + math.exp(-self.lbda_pref * (delta_util)))
        pba_indif = math.exp(-self.lbda_indif * abs(delta_util))

        return pba_pref, pba_indif
