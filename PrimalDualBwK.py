import numpy as np
from scipy.optimize import linprog

from models import (Algorithm, Arm, IID_InputModel)

# make input model only return one cost and reward


class PrimalDualBwK(Algorithm):

    def __init__(self, num_resources, budget, arms, input_model):
        self.resource_consumtion_vector = [0] * num_resources
        self.resource_vector = [0] * num_resources
        self.budget = budget
        self.epsilon = np.sqrt(np.log(len(self.resource_vector)) / budget)
        self.C_rad = np.log(num_resources * budget * len(arms))
        for arm in arms:
            arm.curr_cost_vector = [0] * num_resources
            arm.expected_cost_vector = [0] * num_resources
            arm.expected_reward = 0
        super(PrimalDualBwK, self).__init__(arms, input_model)

    def run(self):
        time = 0

        # Initialization
        for i, arm in enumerate(self.arms):
            arm.pull_arm()
            self.update_arm(arm, i)
            self.update_resource_consumption(arm)
            # get vector of reward and costs
            # increment num pulls
            self.total_reward += arm.curr_reward
            time += 1

        # solve LP problem
        self.resource_vector = self.solve_LP()

        while not self.exhausted():
            arm, l = self.calc_max_bpb()
            arm.pull_arm()
            # update estimated unit cost for each resource i
            self.update_arm(arm, arm.index)
            self.update_resource_consumption(arm)
            self.total_reward += arm.curr_reward
            self.update_resource_vector(l)
            time += 1

    def calc_UCB_reward(self, arm):
        rad = np.sqrt((self.C_rad * arm.expected_reward) /
                      arm.num_pulls) + (self.C_rad / arm.num_pulls)
        return arm.expected_reward + rad

    def update_arm(self, arm, index):
        # updates the current reward and current costs, and also updates the
        # expected reward and expected costs
        reward, costs = self.input_model.get_new_input(index)
        arm.curr_reward = reward
        arm.expected_reward = (
            (arm.expected_reward * (arm.num_pulls - 1)) + reward) / arm.num_pulls
        arm.curr_cost_vector = costs
        for i, cost in enumerate(costs):
            arm.expected_cost_vector[i] = ((arm.expected_cost_vector[i] * (arm.num_pulls - 1)) + cost) / arm.num_pulls

    def calc_LCB_costs(self, arm):
        l = []
        for cost in arm.expected_cost_vector:
            rad = np.sqrt((self.C_rad * cost) / arm.num_pulls) + (self.C_rad / arm.num_pulls)
            l.append(cost - rad)
        # return LCB for resource consumption vector
        return l

    def update_resource_consumption(self, arm):
        # update the resources consumed thus far
        for i, cost in arm.curr_cost_vector:
            self.resource_consumtion_vector[i] += cost

    def update_resource_vector(self, l):
        for i, resource in enumerate(self.resource_vector):
            self.resource_vector[i] = self.resource_vector[
                i] * np.power((1 + self.epsilon), l[i])

    def exhausted(self):
        for val in self.resource_consumtion_vector:
            if val <= 0:
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
            A.append([x * -1 for x in arm.expected_cost_vector])
            b.append(-1 * arm.expected_reward)
        bounds = []
        for i in range(0, len(self.resource_consumtion_vector)):
            bounds.append((0, None))
        res = linprog(c, A, b, bounds=bounds)
        return res.x
