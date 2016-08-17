import numpy as np


class Arm(object):

    def __init__(self, index):
        self.reward = 0
        self.cost = 0
        self.num_pulls = 0
        self.curr_time = 0
        self.index = index

    # need to figure out what e stands for
    def calc_e(self):
        return np.sqrt((2 * np.log(self.curr_time)) / self.num_pulls)

    # is this unique to each arm?
    def calc_beta(self):
        pass

    # is this unique to each arm?
    def calc_lambda(self):
        pass


class Algorithm(object):

    # need to figure out inputs to algorithms

    # each algorithm should have a set of arms
    # should initialize all the arms
    def __init__(self, num_arms):
        self.arms = []
        for i in range(0, num_arms):
            self.arms.append(Arm(i))

    # each algorithm should have a run method
    def run(self):
        pass

    # each algorithm needs to be able to pull the arm
    def pull_arm(self, arm_index):
        arm = self.arms[arm_index]
        # pull the arm

    # @property
    # def reward(self):
    #     return self._reward

    # @reward.setter
    # def reward(self, value):
    #     self._reward = value

    # @property
    # def cost(self):
    #     return self._cost

    # @cost.setter
    # def cost(self, value):
    #     self._cost = value

    # @property
    # def num_pulls(self):
    #     return self._num_pulls

    # @num_pulls.setter
    # def num_pulls(self, value):
    #     self._num_pulls = value

    # @property
    # def curr_time(self):
    #     return self._curr_time

    # @curr_time.setter
    # def curr_time(self, value):
    #     self._curr_time = value

    # @property
    # def index(self):
    #     return self._index

    # @index.setter
    # def index(self, value):
    #     self._index = value
