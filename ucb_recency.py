import numpy as np

from models import (UCB, Arm, IID_InputModel)


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
        curr_sample_cost = 0
        curr_update_cost = 0

        time = 0

        # start by randomly picking an arm
        curr_arm = self.sample_arm()
        for arm in self.arms:
            arm.pull_arm()
            self.update_arm(arm)
            # self.total_reward += arm.curr_reward
            time += 1

        curr_arm = self.calc_max_UCB()
        # arm with highest UCB, to be selected during update steps
        max_arm = curr_arm

        sampled = False

        # keeps track of the current sampling and updating rates
        s_rate = self.sampling_rate
        u_rate = self.update_rate
        while(self.total_cost <= self.budget):
            # checks to see if last step was sampling, if so, then switch back
            # to max arm
            if sampled:
                curr_arm = max_arm

            # sampling time
            if round(curr_sample_cost, 1) == round(s_rate, 1):
                curr_sample_cost = 0
                curr_arm = self.sample_arm()
                sampled = True
                # reset sampling rate
                s_rate = np.random.exponential(scale=1 / self.sampling_rate)

            # update time
            elif round(curr_update_cost, 1) == round(u_rate, 1):
                curr_update_cost = 0
                curr_arm = self.calc_max_UCB()
                max_arm = curr_arm
                self.reset_all_arms()
                # reset updating rate
                u_rate = np.random.exponential(scale=1 / self.update_rate)

            # pull the current arm
            curr_arm.pull_arm()
            # update all the arms with the new inputs
            self.update_arm(curr_arm)

            # update the sampling costs and updating costs
            curr_sample_cost += curr_arm.curr_cost
            curr_update_cost += curr_arm.curr_cost

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

    def sample_arm(self):
        return np.random.choice(self.arms)


# inputs = IID_InputModel()
# arm_one = Arm(0)
# arm_two = Arm(1)
# recency = UCB_Recency(8, [arm_one, arm_two], 2, inputs)
# recency.run()
