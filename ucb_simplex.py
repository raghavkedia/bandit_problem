import numpy as np

from models import (UCB, Arm, IID_InputModel)


class UCB_Simplex(UCB):

    def __init__(self, budget, arms, beta, input_model):
        self.budget = budget
        self.beta = beta
        super(UCB_Simplex, self).__init__(arms, input_model)

    # should override run method
    def run(self):
        time = 0
        while(self.total_cost <= self.budget):
            if time == 0:
                max_arm = self.arms[0]
            else:
                max_arm = self.calc_max_UCB()
            max_arm.pull_arm()
            self.update_arms()
            self.total_cost += max_arm.curr_cost
            self.total_reward += max_arm.curr_reward
            print 'total_cost: {0}, total_reward: {1}, arm_pulled: {2}'.format(
                self.total_cost, self.total_reward, max_arm.index)
            time = time + 1

    def update_arms(self):
        costs, rewards = self.input_model.get_new_input(self.arms)

        for i, arm in enumerate(self.arms):
            arm.cost = ((arm.cost * arm.curr_time) +
                        costs[i]) / (arm.curr_time + 1)

            arm.reward = ((arm.reward * arm.curr_time) +
                          rewards[i]) / (arm.curr_time + 1)

            arm.curr_cost = costs[i]
            arm.curr_reward = rewards[i]
            arm.curr_time = arm.curr_time + 1

        # print self.arms[0].cost
        # print self.arms[0].reward


inputs = IID_InputModel()
arm_one = Arm(0)
arm_two = Arm(1)
simplex = UCB_Simplex(50, [arm_one, arm_two], 2, inputs)
simplex.run()
