import numpy as np

from models import (Algorithm, Arm, IID_InputModel)


class PrimalDualBwK(Algorithm):

    def __init__(self, resource_vector, budget, arms, input_model):
        self.resource_vector = resource_vector
        self.budget = budget
        self.epsilon = np.sqrt(np.log(len(resource_vector)) / budget)
        super(PrimalDualBwK, self).__init__(arms, input_model)

    def run(self):
        time = 0

        # Initialization
        for arm in self.arms:
            # pull each arm once
            # get vector of reward and costs
            # increment num pulls
            time += 1

        # solve LP problem

        while not self.exhausted():
            arm, l = self.calc_max_bpb()
            arm.pull_arm()
            # update estimated unit cost for each resource i
            self.update_arms()
            self.update_resource_vector(l)
            time += 1

    def calc_UCB_reward(self, arm):
        rad = np.sqrt((self.C_rad * arm.avg_reward) /
                      arm.num_pulls) + (self.C_rad / arm.num_pulls)
        return arm.avg_reward + rad

    def calc_LCB_costs(self, arm):
        # return LCB for resource consumption vector
        pass

    def update_resource_vector(self, l):
        for i, resource in enumerate(self.resource_vector):
            self.resource_vector[i] = self.resource_vector[
                i] * np.power((1 + self.epsilon), l[i])

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
