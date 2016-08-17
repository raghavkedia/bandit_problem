from .models import (Algorithm, Arm)


class UCB_Simplex(Algorithm):

    def __init__(self, budget, num_arms):
        self.budget = budget
        super(UCB_Simplex, self).__init__(num_arms)

    # should override run method
    def run(self):
        pass

    def calc_max_UCB(self):
        max_index = 0
        max_UCB = calc_UCB(self.arms[0])
        for index, arm in enumerate(self.arms[1:]):
            ucb = calc_UCB(arm)
            if ucb > max_UCB:
                max_index = index
                max_UCB = ucb

        return max_index

    def calc_UCB(self, arm):
        # ( r_{kt} + beta * e_{kt} ) / c_{kt}
