from __future__ import division

import numpy as np
from scipy.optimize import linprog
from models import (Algorithm, Arm)



class PrimalDualBwK(Algorithm):

    def __init__(self, num_resources, budget, lam, arms, input_model):
        self.resource_consumtion_vector = [0] * num_resources
        self.resource_vector = [0] * num_resources
        self.lam = lam
        self.budget = budget
        # self.epsilon = np.sqrt(np.log(num_resources) / budget)
        self.epsilon = np.sqrt(1 / budget)
        self.C_rad = np.log(num_resources * (budget/lam) * len(arms))
        for arm in arms:
            arm.curr_costs = [0] * num_resources
            arm.expected_costs = [0] * num_resources
        super(PrimalDualBwK, self).__init__(arms, input_model)

    def run(self):
        time = 0

        # Initialization
        for i, arm in enumerate(self.arms):
            arm.pull_arm()
            self.input_model.update_arms(arm, self.arms)
            self.update_stats(arm)
            self.update_resource_consumption(arm)
            # get vector of reward and costs
            # increment num pulls
            # self.total_reward += arm.curr_reward
            time += 1

        # solve LP problem
        self.resource_vector = self.solve_LP()

        while not self.exhausted():
            arm, l = self.calc_max_bpb()
            arm.pull_arm()
            self.input_model.update_arms(arm, self.arms)
            self.update_stats(arm)
            self.update_resource_consumption(arm)
            self.update_resource_vector(l)
            time += 1

        return self.total_reward

    def calc_UCB_reward(self, arm):
        rad = np.sqrt((self.C_rad * arm.expected_reward) /
                      arm.num_pulls) + (self.C_rad / arm.num_pulls)
        return min(arm.expected_reward + rad, 1)

    def calc_LCB_costs(self, arm):
        l = []
        for cost in arm.expected_costs:
            rad = np.sqrt((self.C_rad * cost) / arm.num_pulls) + \
                (self.C_rad / arm.num_pulls)
            l.append(max(self.lam, cost - rad))
        # return LCB for resource consumption vector
        return l

    def update_resource_consumption(self, arm):
        # update the resources consumed thus far
        for i, cost in enumerate(arm.curr_costs):
            self.resource_consumtion_vector[i] += cost

    def update_resource_vector(self, l):
        for i, resource in enumerate(self.resource_vector):
            self.resource_vector[i] = self.resource_vector[
                i] * np.power((1 + self.epsilon), l[i])

    # checks to see if any of the resources have been exhausted
    def exhausted(self):
        for val in self.resource_consumtion_vector:
            if val >= self.budget:
                return True
        return False

    def calc_max_bpb(self):
        # iterate through all the arms
        max_arm = None
        max_bpb = 0
        max_l = None
        for i, arm in enumerate(self.arms):

            u = self.calc_UCB_reward(arm)

            l = self.calc_LCB_costs(arm)

            est_cost = np.dot(l, self.resource_vector)
            bpb = u / est_cost

            if i == 0:
                max_arm = arm
                max_bpb = bpb
                max_l = l

            else:
                if bpb > max_bpb:
                    max_arm = arm
                    max_bpb = bpb
                    max_l = l

        return (max_arm, max_l)

    def solve_LP(self):
        c = [self.budget] * len(self.resource_consumtion_vector)
        A = []
        b = []
        for arm in self.arms:
            A.append([max(x, .0001) * -1 for x in arm.expected_costs])
            b.append(-1 * max(arm.expected_reward, .0001))

        bounds = []
        for i in range(0, len(self.resource_consumtion_vector)):
            bounds.append((0, None))
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, options=dict(bland=True, tol=1e-8))
        return res.x
