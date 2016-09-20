from __future__ import division

import numpy as np
from models import (UCB)


class UCB_Recency(UCB):

    def __init__(self, budget, arms, lam, input_model):
        self.budget = budget
        self.lam = lam
        self.sampling_rate = np.power(budget, -(1. / 3.)) / len(arms)
        self.update_rate = np.power(budget, -(2. / 3.))
        super(UCB_Recency, self).__init__(arms, input_model)

    def run(self):

        # Initialization step
        self.time = 1
        initialized = False
        while(not initialized):

            count = 0
            for arm in self.arms:
                if arm.cost == 0:
                    arm.pull_arm()
                    self.input_model.update_arms(arm, self.arms)
                    self.update_stats(arm)
                else:
                    count += 1

            if count == 2:
                initialized = True
            count = 0

        curr_arm = self.calc_max_UCB()
        max_arm = curr_arm
        sampled = False
        
        curr_sample_budget = np.random.exponential(scale=1. / self.sampling_rate)
        curr_update_budget = np.random.exponential(scale=1. / self.update_rate)
       
        while(self.total_cost <= self.budget):
            # checks to see if last step was sampling, if so, then switch back
            # to max arm
            if sampled:
                curr_arm = max_arm

            # sampling time
            if curr_sample_budget <= 0:
                curr_arm = self.sample_arm()
                sampled = True
                # reset sampling rate
                curr_sample_budget = np.random.exponential(scale=(1. / self.sampling_rate))

            # update time
            elif curr_update_budget <= 0:
                curr_arm = self.calc_max_UCB()
                max_arm = curr_arm
                self.reset_all_arms()
                # reset updating rate
                curr_update_budget = np.random.exponential(scale=(1. / self.update_rate))

            # pull the current arm
            curr_arm.pull_arm()
            # update all the arms with the new inputs
            self.input_model.update_arms(curr_arm, self.arms)

            # update the sampling costs and updating costs
            curr_sample_budget -= curr_arm.curr_cost
            curr_update_budget -= curr_arm.curr_cost

            # update total cost and total reward
            self.update_stats(curr_arm)

            self.time = self.time + 1

        return self.total_reward

    def reset_all_arms(self):
        for arm in self.arms:
            arm.reward = 0
            arm.cost = 0
            arm.curr_cost = 0
            arm.curr_reward = 0
            arm.num_pulls = 1

    def sample_arm(self):
        return self.arms[np.random.randint(0, 2)]

