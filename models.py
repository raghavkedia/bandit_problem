from __future__ import division

import numpy as np


# class for arm objects
class Arm(object):

    def __init__(self, index, reward_range=None, costs_ranges=None):
        self.reward_range = reward_range
        self.costs_ranges = costs_ranges
        self.reward = 0
        self.cost = 0
        self.curr_cost = 0
        self.curr_reward = 0
        # I'm initializing to 1 just so we dont divide by 0 when calculating
        self.num_pulls = 1
        self.curr_time = 0
        self.index = index

    # From the confidence interval
    def calc_e(self, time):
        return np.sqrt((2 * np.log(time)) / self.num_pulls)

    def pull_arm(self):
        self.num_pulls = self.num_pulls + 1

# For these arms, there is a probability on whether theres a reward OR a cost
class Probability_Arm(Arm):

    def __init__(self, index, probability, reward_range, costs_ranges):
        self.probability = probability
        super(Probability_Arm, self).__init__(index=index,
                                              reward_range=reward_range, costs_ranges=costs_ranges)

# make a new markov arm that takes in a reward and a function in the state distributions matrix.
# add comments of different arms and input models

class Markov_Arm(Arm):

    def __init__(self, index, state, state_distributions, state_transition_matrix):
        self.state = state
        self.state_distributions = state_distributions
        self.state_transition_matrix = state_transition_matrix
        super(Markov_Arm, self).__init__(
            index=index, reward_range=None, costs_ranges=None)


# Base class for all input models.
class InputModel(object):

    # override for each input model
    # This method should return two vectors: (costs, rewards)
    def get_new_input(self, arms):
        pass

    # override for each input model
    # This method should make all the necessary updates to the current arm being pulled, and all the other arms. 
    def update_arms(self, curr_arm, arms):
        pass

# IID input model fo bandit, only use with IID arms
class IID_InputModel_Bandit(InputModel_Bandit):

    def __init__(self):
        super(IID_InputModel_Bandit, self).__init__()

    def get_new_input(self, arm):
        reward = np.random.normal(arm.reward_range, (1. / 4000), 1)[0]
        reward = max(0, min(reward, 20)) / 20
        costs = []
        for cost_range in arm.costs_ranges:
            # fixed cost, not from distribution
            cost = cost_range
            costs.append(cost)
        return reward, costs

# Only use for Probability Arms
class Probability_InputModel_Bandit(InputModel_Bandit):

    # ONLY INTENDED FOR PROBABILITY ARMS
    def __init__(self):
        super(Probability_InputModel_Bandit, self).__init__()

    def get_new_input(self, arm):
        value = np.random.random_sample()
        # return a reward
        if value <= arm.probability:
            reward = np.random.normal(arm.reward_range, (1. / 4000), 1)[0]
            reward = max(0, min(reward, 20)) / 20
            costs = [0] * len(arm.costs_ranges)
            return reward, costs
        else:
            costs = []
            for cost_range in arm.costs_ranges:
                # fixed cost, not from distribution
                cost = cost_range
                costs.append(cost)
            return 0, costs

# Only use for Markov Arms
class Markov_InputModel_Bandit(InputModel_Bandit):

    def __init__(self):
        super(Markov_InputModel_Bandit, self).__init__()

    def _get_new_input(self, arm):
        distributions = arm.state_distributions[arm.state]
        # reward = np.random.normal(distributions[0], (1. / 4000), 1)[0]
        # reward = max(0, min(reward, 20)) / 20
        reward = distributions[0]

        # check if there's a function for the costs
        if callable(distributions[1]):
            costs = distributions[1](reward)
        else:
            costs = []
            for cost_range in distributions[1]:
                # fixed cost, not from distribution
                cost = cost_range
                costs.append(cost)
        
        return reward, costs

    def update_arms(self, curr_arm, arms):
        self._update_states(arms)
        self._update_arm(curr_arm)

    def _update_arm(self, arm):
        reward, costs = self._get_new_input(arm)
        arm.cost = ((arm.cost * arm.num_pulls) +
                    costs[0]) / (arm.num_pulls + 1)

        arm.reward = ((arm.reward * arm.num_pulls) +
                      reward) / (arm.num_pulls + 1)

        arm.curr_cost = costs[0]
        arm.curr_reward = reward
        arm.curr_time = arm.curr_time + 1

    def _update_states(self, arms):
        for arm in arms:
            new_state = np.random.choice(np.arange(
                0, len(arm.state_distributions)), p=arm.state_transition_matrix[arm.state])
            arm.state = new_state


# Only use with IID arms, not bandit setting. Used for testing
class IID_InputModel(InputModel):

    def __init__(self):
        # by range, I mean the max possible cost or reward for a given arm.
        # for example, for arm 0, it's costs can range from 0 to .1, and rewards from 0 to 1
        # for arm 1, it's costs can range from 0 to .1, and rewards from 0 to 2
        # the index in the lists correspond to arms with the same index
        self.cost_ranges = [.1, .1]
        self.reward_ranges = [1, 2]
        super(IID_InputModel, self).__init__()

    # overriding
    def get_new_input(self, arms):

        if len(arms) != len(self.cost_ranges):
            # THROW AN ERROR
            return

        costs = [0] * len(arms)
        rewards = [0] * len(arms)
        # make the costs and rewards vectors for the new inputs
        for i in range(0, len(self.cost_ranges)):
            costs[i] = self.cost_ranges[i] * np.random.random_sample()
            rewards[i] = self.reward_ranges[i] * np.random.random_sample()

        return (costs, rewards)


# Base class for all algorithms
class Algorithm(object):

    # each algorithm should have a set of arms
    def __init__(self, arms, input_model):
        self.time = 0
        self.arms = arms
        self.total_cost = 0
        self.total_reward = 0
        self.input_model = input_model

    # each algorithm should have a run method
    def run(self):
        pass

    def update_stats(self, arm):
        self.total_cost += arm.curr_cost
        self.total_reward += arm.curr_reward



# Base class for UCB algorithms (Simplex and Recency)
class UCB(Algorithm):

    def calc_max_UCB(self):
        max_arm = self.arms[0]
        max_UCB = self.calc_UCB(self.arms[0])
        for arm in self.arms[1:]:
            ucb = self.calc_UCB(arm)
            if ucb > max_UCB:
                max_arm = arm
                max_UCB = ucb
        return max_arm

    def calc_UCB(self, arm):
        beta = 1 + (1 / (self.lam))
        numerator = arm.reward + (beta * arm.calc_e(self.time))
        return numerator / arm.cost
