from __future__ import division

import matplotlib.pyplot as plt
from PrimalDualBwK import PrimalDualBwK
from ucb_simplex import UCB_Simplex
from ucb_recency import UCB_Recency
from opt_algo import OPT
from models import (Markov_InputModel_Bandit, Markov_Arm)


# resets all the arms, so they can be fed into another algorithm
def reset_all_arms(arms):
    for arm in arms:
        arm.expected_reward = 0
        arm.expected_costs = [0]
        arm.curr_costs = [0]
        arm.curr_reward = 0
        arm.num_pulls = 0
        arm.state = 0


# this function runs the simulations.
# it will first run for the simplex algo, the the recency algo, and then
# the BwK algo.

def run_simulations(arms, input_model, max_cost, min_budget, max_budget, budget_increment):
    results = []
    for budget in range(min_budget, max_budget + budget_increment, budget_increment):

        print budget

        # Comment out the below three lines if you do not want regret, and just want reward. 
        opt = OPT(budget / max_cost, arms, input_model)
        index, opt_reward = opt.run()
        reset_all_arms(arms)

        print opt_reward

        simplex = UCB_Simplex(budget / max_cost, arms, .1, input_model)
        simplex_reward = simplex.run()
        reset_all_arms(arms)

        print simplex_reward

        recency = UCB_Recency(budget / max_cost, arms, .1, input_model)
        recency_reward = recency.run()
        reset_all_arms(arms)

        print recency_reward

        # bwk = PrimalDualBwK(1, budget / max_cost, .1, arms, input_model)
        # bwk_reward = bwk.run()
        # reset_all_arms(arms)

        # print bwk_reward
        print "====="

        # NOTE: This will print the regret. If you want to print total_reward, 
        # remove the subtraction from the opt_reward. 
        result = [budget / max_cost, opt_reward - simplex_reward,
                  opt_reward - recency_reward]
        results.append(result)

    return results

# this function makes the plots


def make_plots(results, titles):
    plt.close('all')
    f, axarr = plt.subplots(len(results), sharex=False)
    f.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top='off',
                    bottom='off', left='off', right='off')
    plt.xlabel("Budget")
    # RENAME TO TOTAL REWARD IF YOU WANT TO PLOT REWARD
    plt.ylabel("Total Regret")

    for i, result in enumerate(results):
        x = zip(*result)[0]
        simplex = zip(*result)[1]
        recency = zip(*result)[2]
        # bwk = zip(*result)[3]
        axarr[i].plot(x, simplex, 'rs-', label="Simplex")
        axarr[i].plot(x, recency, 'go-', label="Recency")
        # axarr[i].plot(x, bwk, 'yd-', label="BwK")
        # axarr[0].ylabel('Total Reward')
        # axarr[0].xlabel('Budget')
        axarr[i].set_title(titles[i])

    axarr[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

# Here is where you can run your simulations. Each simulation will run for Simplex, Recency, and BwK

# Here is a very basic example. Let's say we want to run an simulation for two arms. Each arm will have two states. 
# Lets say for arm_one, in the first state, the reward is 1, and the cost is .1, and in the second state, the reward
# is .1, and the cost is 1. And let's say that the transition probability from state 0 to state 1 is .5, the vice a versa. 
# We would define arm_one as follows:

# this is the array that contains the reward and cost for each state. Each element in this array corresponds to a state, and is
# in the form [reward, [cost]]
# state_dist_one = [[1.0, [.1]], [.1, [1]]]
# this is the array that contains the state transition probabilites. Each element corresponds to a state (index of element is the state)
# Each element is of the form [prob_state_0, prob_state_1, ....., prob_state_n]. Here we only have two states, so we only need a list of length two
# state_transition_one = [[.5, .5], [.5, .5]]
# Here we define the actual arm object. The first parameter is the index of the arm. The second parameter is the initial state of the arm. And the 
# last two parameters are the two arrays that describe the states and state transitions of the arm. 
# arm_one = Markvov_Arm(0, 0, state_dist_one, state_transition_one)

# Let's define arm_two similarly, just with slightly different numbers:

# state_dist_two = [[0.9, [.1]], [.1, [.9]]]
# state_transition_two = [[.999, .001], [0.001, .999]]
# arm_two = Markov_Arm(1, 0, state_dist_two, state_transition_two)

# Now that we have both our arms, it's time to feed them into our simulation. 
# Before we do that, we do need an input model that's responsible to collecting all the inputs
# and updating the arms. We can define one as such:
# markov_input_model = Markov_InputModel_Bandit()
# we also need to put our arms in a list:
# arms = [arm_one, arm_two]

# we run the simulation as follows:
# result = run_simulations(arms, markov_input_model, 1, 10000, 100000, 10000)
# The first parameter is the list of arms. The second parameter is the input model. The third parameter is the max cost amongst all the arms
# (in this case it's one). The next three parameters are the minimum budget, maximum budget, and the budget increment. 

# once we get our result, we can plot it like so:

# results = [result]
# title of each graph
# titles = ['Simulation Example']
# make_plots(results, titles)

# In the example above, the costs are fixed numbers
# It is also possible to pass in a cost function. 
# For example, define any function that takes in a reward value and outputs a cost:

# def cost_func(reward):
#     return 1 - reward

# you can pass in this function in the same spot the costs would go
# state_dist_temp = [[1.0, [cost_func]], [0.01, [cost_func]]]
# state_transition_temp = [[.999, .001], [.001, .999]]
# arm_temp = Markov_Arm(0, 0, state_dist_one, state_transition_temp)

# Also, right now the reward is a fixed value. If you would like for it to be 
# pull from a distribution, see my note in models.py, in the Markov_InputModel_Bandit class


# Below you will find two simulations that we ran: Simulation A and Simulation B. 

markov_input_model = Markov_InputModel_Bandit()

# Simulation A

state_dist_one = [[1.0, [.01]], [0.01, [1]]]
state_transition_one = [[.999, .001], [.001, .999]]
arm_one = Markov_Arm(0, 0, state_dist_one, state_transition_one)

state_dist_two = [[0.9, [.1]], [.1, [.9]]]
state_transition_two = [[.999, .001], [0.001, .999]]
arm_two = Markov_Arm(1, 0, state_dist_two, state_transition_two)

state_dist_three = [[0.8, [.15]], [.15, [.8]]]
state_transition_three = [[.999, .001], [0.001, .999]]
arm_three = Markov_Arm(2, 0, state_dist_three, state_transition_three)

state_dist_four = [[0.7, [.2]], [.2, [.7]]]
state_transition_four = [[.999, .001], [0.001, .999]]
arm_four = Markov_Arm(3, 0, state_dist_four, state_transition_four)

arms = [arm_one, arm_two, arm_three, arm_four]

# result_A = run_simulations(arms, markov_input_model, 1, 10000, 50000, 10000)

# Simulation B

state_dist_one = [[1, [.0001]], [0.1, [1]]]
state_transition_one = [[.9999, .0001], [.0001, .9999]]
arm_one = Markov_Arm(0, 0, state_dist_one, state_transition_one)

state_dist_two = [[0.01, [1]], [1, [1]]]
state_transition_two = [[.9999, .0001], [0.0001, .9999]]
arm_two = Markov_Arm(1, 0, state_dist_two, state_transition_two)

arms = [arm_one, arm_two]

# result_B = run_simulations(arms, markov_input_model, 1, 10000, 10001, 10000)

# Simulation C

state_dist_one = [[.5, [.5]], [0.5, [.5]]]
state_transition_one = [[1, 0], [1, 0]]
arm_one = Markov_Arm(0, 0, state_dist_one, state_transition_one)

state_dist_two = [[0.1, [1]], [1, [.1]]]
state_transition_two = [[.9999, .0001], [0, 1]]
arm_two = Markov_Arm(1, 0, state_dist_two, state_transition_two)

arms = [arm_one, arm_two]

# result_C = run_simulations(arms, markov_input_model, 1, 10000, 50000, 10000)

# Simulation D

state_dist_one = [[.5, [.5]], [0.5, [.5]]]
state_transition_one = [[1, 0], [1, 0]]
arm_one = Markov_Arm(0, 0, state_dist_one, state_transition_one)

state_dist_two = [[0.1, [1]], [0.9, [.1]]]
state_transition_two = [[.999, .001], [0.01, .99]]
arm_two = Markov_Arm(1, 0, state_dist_two, state_transition_two)

arms = [arm_one, arm_two]

result_D = run_simulations(arms, markov_input_model, 1, 10000, 50000, 10000)

# Simulation E

state_dist_one = [[.5, [.5]], [0.5, [.5]]]
state_transition_one = [[1, 0], [1, 0]]
arm_one = Markov_Arm(0, 0, state_dist_one, state_transition_one)

state_dist_two = [[0.1, [1]], [1.5, [.1]]]
state_transition_two = [[.999, .001], [0.05, .95]]
arm_two = Markov_Arm(1, 0, state_dist_two, state_transition_two)

result_E = run_simulations(arms, markov_input_model, 1, 10000, 50000, 10000)

# Print results here

# pass each result into this array to be printed. Each result will
# correspond to a different graph.
results = [result_D, result_E]
# title of each graph
titles = ['Simulation D', 'Simulation E']

make_plots(results, titles)
