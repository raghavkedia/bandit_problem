from __future__ import division

import numpy as np
from models import (UCB)


class UCB_Recency(UCB):

    def __init__(self, budget, arms, beta, input_model):
        self.budget = budget
        self.beta = beta
        self.sampling_rate = np.power(budget, -(1. / 3.)) / len(arms)
        self.update_rate = np.power(budget, -(2. / 3.))
        super(UCB_Recency, self).__init__(arms, input_model)

    def run(self):

        # these keep track of how much is spent between sampling and updating
        # times
        curr_sample_budget = 0
        curr_update_budget = 0

        time = 0

        # start by randomly picking an arm
        # curr_arm = self.sample_arm()
        for arm in self.arms:
            arm.pull_arm()
            self.update_arm(arm)
            self.total_reward += arm.curr_reward
            self.total_cost += arm.curr_cost
            time += 1

        curr_arm = self.calc_max_UCB()
        # arm with highest UCB, to be selected during update steps
        max_arm = curr_arm

        sampled = False

        # keeps track of the current sampling and updating rates
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
                # curr_sample_budget = 5

            # update time
            elif curr_update_budget <= 0:
                curr_arm = self.calc_max_UCB()
                max_arm = curr_arm
                self.reset_all_arms()
                # reset updating rate
                curr_update_budget = np.random.exponential(scale=(1. / self.update_rate))
                # curr_update_budget = 20

            # pull the current arm
            curr_arm.pull_arm()
            # update all the arms with the new inputs
            self.input_model.update_states(self.arms)
            self.update_arm(curr_arm)

            # update the sampling costs and updating costs
            curr_sample_budget -= curr_arm.curr_cost
            curr_update_budget -= curr_arm.curr_cost

            # update total cost and total reward
            self.total_cost += curr_arm.curr_cost
            self.total_reward += curr_arm.curr_reward
            # print 'total_cost: {0}, total_reward: {1}, arm_pulled: {2}'.format(
            #     self.total_cost, self.total_reward, curr_arm.index)
            time = time + 1

        return self.total_reward

    def reset_all_arms(self):
        for arm in self.arms:
            arm.reward = 0
            arm.cost = 0
            arm.curr_cost = 0
            arm.curr_reward = 0
            arm.num_pulls = 1
            # arm.state = 0

    def sample_arm(self):
        return self.arms[np.random.randint(0, 2)]


# inputs = IID_InputModel()
# arm_one = Arm(0)
# arm_two = Arm(1)
# recency = UCB_Recency(8, [arm_one, arm_two], 2, inputs)
# recency.run()
