import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import gym
import planner_utils as utils
import heapq
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tol = 1e-3

dtype = torch.float

env = gym.make('Pendulum-v0')

model = utils.load_model('pendulum_model.pt')


def astar(model, start, goal, discretized_theta, discretized_theta_dot, discretized_action):

	frontier = utils.PriorityQueue()
	frontier.put(start, 0)
	came_from = {}
	cost_so_far = {}
	came_from[start] = None
	cost_so_far[start] = 0

	iterations = 0
	while not frontier.empty():
	    iterations += 1
	    current = frontier.get()
	    # print(iterations,current,cost_so_far[current])

	    if current[0:2] == goal[0:2]:
	        break	    

	    for next in utils.get_possible_states(current,discretized_action, discretized_theta, discretized_theta_dot):
	        new_cost = cost_so_far[current] + 1
	        
	        if next not in cost_so_far or new_cost < cost_so_far[next]:
	            cost_so_far[next] = new_cost
	            priority = new_cost + utils.calculate_h(next,goal)
	            frontier.put(next, priority)
	            came_from[next] = current

	# print("planner iterations = {}".format(iterations))

	# iterations = len(came_from) #frontier.getlength()

	return came_from, cost_so_far, current,iterations


def reconstruct_path(came_from, start, goal):
    current = goal
    actions = []
    path = []
    while current != start:
        path.append(current[0:2])
        actions.append(current[2])
        current = came_from[current]
    path.append(start[0:2]) # optional
    actions.append(start[2])
    actions.reverse()
    path.reverse() # optional
    return path,actions


def main(num_action_discretizations,num_state_discretizations):

	discretized_theta, discretized_theta_dot, discretized_action = utils.discretize_state_action(num_action_discretizations, num_state_discretizations)
	env.seed(0)
	observation = env.reset()
	# print(observation)
	initial_state = np.array([np.arctan2(observation[1],observation[0]),observation[2]]).T
	# initial_state = np.array([1.2, 0.0])
	initial_state = utils.find_closest_disc_state(initial_state,discretized_theta,discretized_theta_dot)
	initial_state = initial_state + (0.,)
	# print(initial_state)

	goal_state = np.array([0.0,0.0])
	goal_state = utils.find_closest_disc_state(goal_state,discretized_theta,discretized_theta_dot)
	goal_state = goal_state + (0.,)

	came_from, cost_so_far,goal,iterations = astar(model,initial_state,goal_state, discretized_theta, discretized_theta_dot, discretized_action)
	
	path,planned_actions = reconstruct_path(came_from,initial_state,goal)

	# print(len(planned_actions))
	


	# for i,action in enumerate(planned_actions):
	# 	print(i,observation)
	# 	env.render()
	# 	observation, reward, done, info = env.step([action])
		
	# env.close()	

	return iterations


if __name__ == '__main__':
	
	iter_list=[]
	ranges = np.arange(11,151,10)
	for disc in ranges:
		iterations = main(num_action_discretizations=20,num_state_discretizations=disc)
		print(iterations,disc)
		iter_list.append(iterations)
		

	plt.ylabel('Iterations by Planner')
	plt.xlabel('Discretization resolution')
	# plt.title('Action Space Discretization with constant state discretization=31')
	plt.title('State Space Discretization with constant action discretization=20')
	plt.grid(True)
	plt.plot(ranges, np.array(iter_list))
	plt.show() 