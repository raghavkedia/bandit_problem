import numpy as np

from models import (UCB, Arm, InputModel)


class UCB_Recency(Algorithm):

    def __init__(self, budget, arms, beta):
        self.budget = budget
        self.beta = beta
        self.sampling_rate = np.power(budget, -(1 / 3)) / len(arms)
        self.update_rate = np.powser(budget, -(2 / 3))
        super(UCB_Recency, self).__init__(arms)

    def run(self):
        curr_sample_cost = 0
        curr_update_cost = 0
        time = 0
        curr_arm = sample_arm()
        max_arm = None
        sampled = False
        s_rate = self.sampling_rate
        u_rate = self.update_rate
        while(self.total_cost <= self.budget):

            if sampled:
                curr_arm = max_arm
            # sampling time
            if round(curr_sample_cost, 3) == round(s_rate, 3):
                curr_sample_cost = 0
                curr_arm = sample_arm()
                sampled = True
                s_rate = np.random.exponential(scale=1 / self.sampling_rate)

            # update time
            elif round(curr_update_cost, 3) == round(u_rate, 3):
                curr_update_cost = 0
                curr_arm = self.calc_max_UCB()
                max_arm = curr_arm
                self.reset_all_arms()
                U_rate = np.random.exponential(scale=1 / self.update_rate)

            curr_arm.pull_arm()
            self.update_arms()
            curr_sample_cost += curr_arm.curr_cost
            curr_update_cost += curr_arm.curr_cost
            self.total_cost += curr_arm.curr_cost
            self.total_reward += cur_arm.curr_reward
            time = time + 1

    def reset_all_arms(self):
        for arm in self.arms:
            arm.reward = 0
            arm.cost = 0
            arm.curr_cost = 0
            arm.curr_reward = 0
            arm.num_pulls = 0

    def sample_arm(self):
        return np.random.choice(self.arms)

    def update_arm(self):
