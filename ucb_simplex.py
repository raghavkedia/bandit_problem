from models import (UCB, Arm, IID_InputModel)


## MAKE BANDIT, ONLY UPDATE EMPIRICAL VALUE WHEN ARM IS PULLED

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
                # just pick the first arm at t = 0
                max_arm = self.arms[0]
            else:
                # get arm with highest ucb
                max_arm = self.calc_max_UCB()

            max_arm.pull_arm()
            # get the new inputs for all the arms
            self.update_arms()
            self.total_cost += max_arm.curr_cost
            self.total_reward += max_arm.curr_reward
            print 'total_cost: {0}, total_reward: {1}, arm_pulled: {2}'.format(
                self.total_cost, self.total_reward, max_arm.index)
            time = time + 1


inputs = IID_InputModel()
arm_one = Arm(0)
arm_two = Arm(1)
simplex = UCB_Simplex(50, [arm_one, arm_two], 2, inputs)
simplex.run()
