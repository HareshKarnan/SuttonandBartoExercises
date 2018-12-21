#######################################################################
# Copyright (C)                                                       #
# 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                  #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# 2017 Aja Rangaswamy (aja004@gmail.com)                              #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
# ^^ code modified from the above source : ^^


import numpy as np
from math import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# some definitions
max_car_move = 5
max_cars = 20
discount = 0.9
rental_credit = 10
move_car_cost = 2
# poisson distribution has infinite bounds. Hence, we need to set an upper bound for the Expectation calculations
POISSON_UPPER_BOUND = 11
RENTAL_REQUEST_FIRST_LOC = 3
RENTAL_REQUEST_SECOND_LOC = 4
RETURNED_FIRST_LOC = 3
RETURNED_SECOND_LOC = 2
DISCOUNT=0.9

actions = np.arange(-max_car_move,max_car_move+1)
stateVal = np.zeros((max_cars+1,max_cars+1))


# function to calculate the probability of event x happening, that follows a poisson distribution
def poisson(x,lam):
	# TODO [Haresh]: can use a dictionary to speed up this function execution (memoization)
	return exp(-lam)*pow(lam,x)/factorial(x)

# function that returns the expected reward for being in a state and taking a particular action.
# state : [# of the cars in the first location, # of cars in the second location]
# action : [-5,-4,....0,1,...4,5] 3 of cars to be moved from one location to another
# stateValue : state value matrix

def expectedreward(state,action,stateValue):
	returns = 0
	# returns = cost per car * number of cars
	returns-= move_car_cost*abs(action)

	# go through all possible rental requests
	for rental_request_first_loc in range(POISSON_UPPER_BOUND):
		for rental_request_second_loc in range(POISSON_UPPER_BOUND):
			# moving cars
			num_of_cars_first_loc = state[0]-action
			num_of_cars_second_loc = state[1]-action
			# ^^^^ this is the state at the next time step for the current action

			# rental requests must be less than the number of cars
			real_rental_first_loc = rental_request_first_loc
			real_rental_second_loc = rental_request_second_loc

			# get credits for renting
			reward = (real_rental_first_loc+real_rental_second_loc) * rental_credit

			# fix the number of cars in each location after moving the cars
			num_of_cars_first_loc-=real_rental_first_loc
			num_of_cars_second_loc-=real_rental_second_loc

			# probability of current combination of rental requests
			prob = poisson(rental_request_first_loc,RENTAL_REQUEST_FIRST_LOC) * poisson(rental_request_second_loc,RENTAL_REQUEST_SECOND_LOC)

			for returned_cars_first_loc in range(POISSON_UPPER_BOUND):
				for returned_cars_second_loc in range(POISSON_UPPER_BOUND):
					num_of_cars_first_loc = min(num_of_cars_first_loc + returned_cars_first_loc,max_cars)
					num_of_cars_second_loc = min(num_of_cars_second_loc + returned_cars_second_loc,max_cars)

					prob_ = poisson(returned_cars_first_loc,RETURNED_FIRST_LOC) * poisson(returned_cars_second_loc,RETURNED_SECOND_LOC)*prob
					returns+=prob_*(reward+DISCOUNT*stateValue[num_of_cars_first_loc,num_of_cars_second_loc])

	return returns

def jacks_car_rental():
	value = np.zeros((max_cars+1,max_cars+1))
	policy = np.zeros(value.shape,dtype=np.int)

	iterations = 0
	_,axes = plt.subplots(2,3,figsize=(40,20))
	plt.subplots_adjust(wspace=0.1,hspace=0.2)
	axes = axes.flatten()

	while True:
		fig = sns.heatmap(np.flipud(policy), cmap="YlGnBu", ax=axes[iterations])
		fig.set_ylabel('# cars at first location', fontsize=30)
		fig.set_yticks(list(reversed(range(max_cars + 1))))
		fig.set_xlabel('# cars at second location', fontsize=30)
		fig.set_title('policy %d' % (iterations), fontsize=30)

		# policy evaluation (in-place)
		while True:
			new_value = np.copy(value)
			for i in range(max_cars+1):
				for j in range(max_cars+1):
					new_value[i,j] = expectedreward([i,j],policy[i,j],new_value)
			value_change = np.sum(np.abs((new_value-value)))
			print('value change :: ',value_change)
			value = new_value.copy()
			if value_change<1e-4:
				print('Not much change in value.. stopping iteration.. ')
				break


		# policy improvement step
		new_policy = np.copy(policy)
		for i in range(max_cars+1):
			for j in range(max_cars+1):
				action_returns = []
				for action in actions:
					if ((action>=0 and i>=action) or (action<=0 and j>=abs(action))):
						action_returns.append(expectedreward([i,j],action,value))
					else:
						action_returns.append(-float('inf'))
				new_policy[i,j] = actions[np.argmax(action_returns)]

		policy_change = (new_policy!=policy).sum()
		print('policy changed in %d states' % policy_change)
		policy = new_policy.copy()

		if policy_change == 0:
			fig = sns.heatmap(np.flipud(value), cmap="YlGnBu", ax=axes[-1])
			fig.set_ylabel('# cars at first location', fontsize=30)
			fig.set_yticks(list(reversed(range(max_cars + 1))))
			fig.set_xlabel('# cars at second location', fontsize=30)
			fig.set_title('optimal value', fontsize=30)
			break

		iterations += 1

if __name__=='__main__':
	jacks_car_rental()








