import numpy as np


class Arm(object):

    def __init__(self, index, input_model):
        self.reward = 0
        self.cost = 0
        self.curr_cost = 0
        self.curr_reward = 0
        self.num_pulls = 0
        self.curr_time = 0
        self.index = index
        self.input_model = input_model

    # need to figure out what e stands for
    def calc_e(self):
        return np.sqrt((2 * np.log(self.curr_time)) / self.num_pulls)

    def pull_arm(self):
        self.num_pulls = self.num_pulls + 1


class InputModel(object):

    # override for each input model
    def update_arm(self, arm):
        pass


class Algorithm(object):

    # need to figure out inputs to algorithms

    # each algorithm should have a set of arms
    # should initialize all the arms
    def __init__(self, arms):
        self.arms = arms
        self.total_cost = 0
        self.total_reward = 0

    # each algorithm should have a run method
    def run(self):
        pass

    # each algorithm needs to be able to pull the arm
    def pull_arm(self, arm_index):
        pass
        # pull the arm
