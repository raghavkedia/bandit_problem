import numpy as np

from models import (UCB, Arm, IID_InputModel)


class UCB_Recency(UCB):

    def __init__(self, budget, arms, beta, input_model):
        self.budget = budget
        self.beta = beta
        self.sampling_rate = np.power(budget, -(1. / 3.)) / len(arms)
        self.update_rate = np.power(budget, -(2. / 3.))
        print self.sampling_rate
        print self.update_rate
        super(UCB_Recency, self).__init__(arms, input_model)

    def run(self):
        curr_sample_cost = 0
        curr_update_cost = 0
        time = 0
        curr_arm = self.sample_arm()
        max_arm = None
        sampled = False
        s_rate = self.sampling_rate
        u_rate = self.update_rate
        while(self.total_cost <= self.budget):

            if sampled:
                curr_arm = max_arm
            # sampling time
            if round(curr_sample_cost, 1) == round(s_rate, 1):
                print "SAMPLING"
                curr_sample_cost = 0
                curr_arm = self.sample_arm()
                sampled = True
                s_rate = np.random.exponential(scale=1 / self.sampling_rate)

            # update time
            elif round(curr_update_cost, 1) == round(u_rate, 1):
                print "UPDATING"
                curr_update_cost = 0
                curr_arm = self.calc_max_UCB()
                max_arm = curr_arm
                self.reset_all_arms()
                u_rate = np.random.exponential(scale=1 / self.update_rate)

            curr_arm.pull_arm()
            self.update_arms()
            curr_sample_cost += curr_arm.curr_cost
            curr_update_cost += curr_arm.curr_cost
            self.total_cost += curr_arm.curr_cost
            self.total_reward += curr_arm.curr_reward
            print 'total_cost: {0}, total_reward: {1}, arm_pulled: {2}'.format(
                self.total_cost, self.total_reward, curr_arm.index)
            time = time + 1

    def reset_all_arms(self):
        for arm in self.arms:
            arm.reward = 0
            arm.cost = 0
            arm.curr_cost = 0
            arm.curr_reward = 0
            arm.num_pulls = 1

    def sample_arm(self):
        return np.random.choice(self.arms)

    def update_arms(self):
        costs, rewards = self.input_model.get_new_input(self.arms)

        for i, arm in enumerate(self.arms):
            arm.cost = ((arm.cost * arm.num_pulls) +
                        costs[i]) / (arm.num_pulls + 1)

            arm.reward = ((arm.reward * arm.num_pulls) +
                          rewards[i]) / (arm.num_pulls + 1)

            arm.curr_cost = costs[i]
            arm.curr_reward = rewards[i]
            arm.curr_time = arm.curr_time + 1

inputs = IID_InputModel()
arm_one = Arm(0)
arm_two = Arm(1)
recency = UCB_Recency(8, [arm_one, arm_two], 2, inputs)
recency.run()
