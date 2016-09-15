from __future__ import division

import matplotlib.pyplot as plt
from PrimalDualBwK import PrimalDualBwK
from ucb_simplex import UCB_Simplex
from ucb_recency import UCB_Recency
from models import (IID_InputModel_Bandit, Markov_InputModel_Bandit,
                    Probability_InputModel_Bandit, Arm, Probability_Arm, Markov_Arm)

inputs = IID_InputModel_Bandit()

probability_input_model = Probability_InputModel_Bandit()

markov_input_model = Markov_InputModel_Bandit()


def reset_all_arms(arms):
    for arm in arms:
        arm.reward = 0
        arm.cost = 0
        arm.curr_cost = 0
        arm.curr_reward = 0
        arm.num_pulls = 1
        arm.state = 0


def run_simulations(arms, input_model, max_cost):
    results = []
    for budget in range(100000, 100001):
        simplex = UCB_Simplex(budget / max_cost, arms, 2, input_model)
        simplex_reward = simplex.run()
        reset_all_arms(arms)

        # # # # for arm in arms:
        # # # #     print arm.state

        recency = UCB_Recency(budget / max_cost, arms, 2, input_model)
        recency_reward = recency.run()
        reset_all_arms(arms)

        bwk = PrimalDualBwK(1, budget / max_cost, arms, input_model)
        bwk_reward = bwk.run()
        reset_all_arms(arms)

        result = [budget / max_cost, simplex_reward, recency_reward, bwk_reward]
        # recency_reward, bwk_reward]
        print result
        results.append(result)

    return results


def make_plots(results, titles):
    plt.close('all')
    f, axarr = plt.subplots(len(results), sharex=False)
    f.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top='off',
                    bottom='off', left='off', right='off')
    plt.xlabel("Budget")
    plt.ylabel("Total Reward")

    for i, result in enumerate(results):
        x = zip(*result)[0]
        simplex = zip(*result)[1]
        recency = zip(*result)[2]
        bwk = zip(*result)[3]
        axarr[i].plot(x, simplex, 'rs-', label="Simplex")
        axarr[i].plot(x, recency, 'go-', label="Recency")
        axarr[i].plot(x, bwk, 'yd-', label="BwK")
        # axarr[0].ylabel('Total Reward')
        # axarr[0].xlabel('Budget')
        axarr[i].set_title(titles[i])

    axarr[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


# # SIMULATION A
# # homogeneous arms with expected reward value-cost parameter pairs

# arm_one = Arm(0, 4, [4. / 4])
# arm_two = Arm(1, 4, [4. / 4])
# arm_three = Arm(2, 4, [4. / 4])
# arm_four = Arm(3, 2.7, [3. / 4])
# arm_five = Arm(4, 2.7, [3. / 4])
# arm_six = Arm(5, 2.7, [3. / 4])
# arms = [arm_one, arm_two, arm_three, arm_four, arm_five, arm_six]

# Probability Arms

# arm_one = Probability_Arm(0, 0.5, 4, [4. / 4])
# arm_two = Probability_Arm(1, 0.5, 4, [4. / 4])
# arm_three = Probability_Arm(2, 0.5, 4, [4. / 4])
# arm_four = Probability_Arm(3, 0.5, 2.7, [3. / 4])
# arm_five = Probability_Arm(4, 0.5, 2.7, [3. / 4])
# arm_six = Probability_Arm(5, 0.5, 2.7, [3. / 4])
# arms = [arm_one, arm_two, arm_three, arm_four, arm_five, arm_six]

# Markov Arms

# state_dist_one = [[4, [4. / 4]], [4, [4. / 4]]]
# state_transition_one = [[.5, .5], [.5, .5]]
# arm_one = Markov_Arm(0, 0, state_dist_one, state_transition_one)

# state_dist_two = [[4, [4. / 4]], [4, [4. / 4]]]
# state_transition_two = [[.5, .5], [.5, .5]]
# arm_two = Markov_Arm(1, 0, state_dist_two, state_transition_two)

# state_dist_three = [[4, [4. / 4]], [4, [4. / 4]]]
# state_transition_three = [[.5, .5], [.5, .5]]
# arm_three = Markov_Arm(2, 0, state_dist_three, state_transition_three)

# state_dist_four = [[2.7, [3. / 4]], [2.7, [3. / 4]]]
# state_transition_four = [[.5, .5], [.5, .5]]
# arm_four = Markov_Arm(3, 0, state_dist_four, state_transition_four)

# state_dist_five = [[2.7, [3. / 4]], [2.7, [3. / 4]]]
# state_transition_five = [[.5, .5], [.5, .5]]
# arm_five = Markov_Arm(4, 0, state_dist_five, state_transition_five)

# state_dist_six = [[2.7, [3. / 4]], [2.7, [3. / 4]]]
# state_transition_six = [[.5, .5], [.5, .5]]
# arm_six = Markov_Arm(5, 0, state_dist_six, state_transition_six)

# arms = [arm_one, arm_two, arm_three, arm_four, arm_five, arm_six]


# results_A = run_simulations(arms, markov_input_model, 4)

# # # SIMULATION B
# # # Diverse Arms

# arm_one = Arm(0, 3, [4. / 20])
# arm_two = Arm(1, 2, [4. / 20])
# arm_three = Arm(2, 0.2, [2. / 20])
# arm_four = Arm(3, 0.16, [2. / 20])
# arm_five = Arm(4, 18, [20. / 20])
# arm_six = Arm(5, 18, [16. / 20])
# arms = [arm_one, arm_two, arm_three, arm_four, arm_five, arm_six]

# results_B = run_simulations(arms, inputs, 20)

# SIMULATION C
# Extremely Diverse Arms

# arm_one = Arm(0, 0.44, [5. / 150])
# arm_two = Arm(1, 0.4, [4. / 150])
# arm_three = Arm(2, 0.2, [3. / 150])
# arm_four = Arm(3, .08, [1. / 150])
# arm_five = Arm(4, 14, [120. / 150])
# arm_six = Arm(5, 18, [150. / 150])
# arms = [arm_one, arm_two, arm_three, arm_four, arm_five, arm_six]

# results_C = run_simulations(arms, inputs, 150)

# state_dist_one = [[4, [4. / 4]], [4, [4. / 4]]]
# state_transition_one = [[.5, .5], [.5, .5]]
# arm_one = Markov_Arm(0, 0, state_dist_one, state_transition_one)

state_dist_one = [[0, [1]], [1, [0]], [0.1, [1]]]
state_transition_one = [[.99, .01, 0], [0, .999, .001], [0, 0, 1]]
arm_one = Markov_Arm(0, 0, state_dist_one, state_transition_one)

state_dist_two = [[0.01, [1]], [1, [0.1]]]
state_transition_two = [[.999, .001], [0, 1]]
arm_two = Markov_Arm(1, 0, state_dist_two, state_transition_two)

# BWK DOES BETTER
# state_dist_one = [[2, [1]], [0, [1]]]
# state_transition_one = [[.5, .5], [.5, .5]]
# arm_one = Markov_Arm(0, 0, state_dist_one, state_transition_one)

# state_dist_two = [[2, [1]], [0, [1]]]
# state_transition_two = [[.7, .3], [.7, .3]]
# arm_two = Markov_Arm(1, 0, state_dist_two, state_transition_two)

arms = [arm_one, arm_two]

result = run_simulations(arms, markov_input_model, 1)

# print result

results = [result, result]
titles = ['Simulation A', 'Simulation C']

make_plots(results, titles)
