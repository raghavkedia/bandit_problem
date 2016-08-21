import numpy as np

from models import (UCB, Arm, InputModel)


class IID_InputModel(InputModel):

    def __init__(self, cost_range, reward_range):
        self.cost_range = cost_range
        self.reward_range = reward_range

    # overriding
    def update_arm(self, arm):
        new_cost = self.cost_range * np.random.random_sample()
        new_reward = self.reward_range * np.random.random_sample()

        arm.cost = ((arm.cost * arm.num_pulls) +
                    new_cost) / (arm.num_pulls + 1)
        arm.reward = ((arm.reward * arm.num_pulls) +
                      new_reward) / (arm.num_pulls + 1)
        arm.num_pulls = arm.num_pulls + 1
        arm.curr_cost = new_cost
        arm.curr_reward = new_reward


class UCB_Simplex(UCB):

    def __init__(self, budget, arms, beta):
        self.budget = budget
        self.beta = beta
        super(UCB_Simplex, self).__init__(arms)

    # should override run method
    def run(self):
        time = 0
        while(self.total_cost <= self.budget):
            max_arm = self.calc_max_UCB()
            max_arm.pull_arm()
            self.update_arms()
            self.total_cost += max_arm.curr_cost
            self.total_reward += max_arm.curr_reward
            print 'total_cost: {0}, total_reward: {1}, arm_pulled: {2}'.format(
                self.total_cost, self.total_reward, max_arm.index)
            time = time + 1

    def update_arms(self):
        for arm in self.arms:
            arm.input_model.update_arm(arm)
            arm.curr_time = arm.curr_time + 1


iid_one = IID_InputModel(1, 1)
iid_two = IID_InputModel(1, 1)
arm_one = Arm(0, iid_one)
arm_two = Arm(1, iid_two)
simplex = UCB_Simplex(100, [arm_one, arm_two], 2)
simplex.run()
