import numpy as np


class Arm(object):

    def __init__(self, index):
        self.reward = 0
        self.cost = 0
        self.curr_cost = 0
        self.curr_reward = 0
        self.num_pulls = 1
        self.curr_time = 0
        self.index = index

    # need to figure out what e stands for
    def calc_e(self):
        return np.sqrt((2 * np.log(self.curr_time)) / self.num_pulls)

    def pull_arm(self):
        self.num_pulls = self.num_pulls + 1


class InputModel(object):

    # override for each input model
    def get_new_input(self, arms):
        pass


class IID_InputModel(InputModel):

    def __init__(self):
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
        for i in range(0, len(self.cost_ranges)):
            costs[i] = self.cost_ranges[i] * np.random.random_sample()
            rewards[i] = self.reward_ranges[i] * np.random.random_sample()

        return (costs, rewards)


class Algorithm(object):

    # need to figure out inputs to algorithms

    # each algorithm should have a set of arms
    # should initialize all the arms
    def __init__(self, arms, input_model):
        self.arms = arms
        self.total_cost = 0
        self.total_reward = 0
        self.input_model = input_model

    # each algorithm should have a run method
    def run(self):
        pass

    # each algorithm needs to be able to pull the arm
    def pull_arm(self, arm_index):
        pass
        # pull the arm

    def update_arms(self, time):
        pass
        # update all the arms with new inputs


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
        numerator = arm.reward + (self.beta * arm.calc_e())
        return numerator / arm.cost
