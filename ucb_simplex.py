from __future__ import division

from models import (UCB, Arm, IID_InputModel)
import numpy as np


# Make input model independent of algo (markov)

class UCB_Simplex(UCB):

    def __init__(self, budget, arms, lam, input_model):
        self.budget = budget
        # Lambda is a lower bound on average resource consumtion
        self.lam = lam
        super(UCB_Simplex, self).__init__(arms, input_model)

    # should override run method
    def run(self):

        # Initialization Step
        # Use this for recency also! 
        self.time = 1
        initialized = False
        while(not initialized):
            
            count = 0
            for arm in self.arms:
                if arm.expected_costs[0] == 0:
                    arm.pull_arm()
                    self.input_model.update_arms(arm, self.arms)
                    self.update_stats(arm)
                else:
                    count += 1

            if count == 2:
                initialized = True

            count = 0
            self.time = self.time + 1

        while(self.total_cost <= self.budget):
            max_arm = self.calc_max_UCB()
            max_arm.pull_arm()
            # get the new inputs for all the arms, update the current arm being pulled
            self.input_model.update_arms(max_arm, self.arms)
            self.update_stats(max_arm)
            self.time = self.time + 1

        return self.total_reward
