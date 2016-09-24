from __future__ import division

import numpy as np
from models import Algorithm


class OPT(Algorithm):

    def __init__(self, budget, arms, input_model):
        self.budget = budget
        for arm in arms:
            arm.total_arm_reward = 0
            arm.total_arm_budget = budget
        super(OPT, self).__init__(arms, input_model)

    def run(self):

        while(not self.exhausted()):

            for arm in self.arms:
                if arm.total_arm_budget <= 0:
                    continue
                arm.pull_arm()
                self.input_model.update_arms(arm, self.arms)
                arm.total_arm_reward += arm.curr_reward
                arm.total_arm_budget -= arm.curr_costs[0]

        max_reward = self.arms[0].total_arm_reward
        max_index = self.arms[0].index
        for arm in self.arms:
            reward = arm.total_arm_reward
            if reward > max_reward:
                max_reward = reward
                max_index = arm.index

        return max_index, max_reward

    def exhausted(self):
        for arm in self.arms:
            if arm.total_arm_budget > 0:
                return False
        return True

    def _get_new_input(self, arm):
        distributions = arm.state_distributions[arm.state]
        reward = distributions[0]

        if callable(distributions[1]):
            costs = distributions[1](reward)
        else:
            costs = []
            for cost_range in distributions[1]:
                # fixed cost, not from distribution
                cost = cost_range
                costs.append(cost)
        return reward, costs

    def _update_arm(self, arm):
        self._update_state(arm)
        reward, costs = self._get_new_input(arm)
        arm.curr_costs = costs
        arm.curr_reward = reward
        arm.curr_time = arm.curr_time + 1

    def _update_state(self, arm):
        new_state = np.random.choice(np.arange(
            0, len(arm.state_distributions)), p=arm.state_transition_matrix[arm.state])
        arm.state = new_state
