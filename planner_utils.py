import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import gym
import heapq

tol = 1e-3

dtype = torch.float


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)


def load_model(model_path):

	device = torch.device("cpu")

	D_in, H, D_out = 5, 128, 3

	model = torch.nn.Sequential(
	    torch.nn.Linear(D_in, H, bias=True),
	    torch.nn.ReLU(),
	    torch.nn.Linear(H, D_out, bias=True),
	).to(device)

	model.load_state_dict(torch.load(model_path))

	model.eval()

	return model



def discretize_state_action(num_action_discretizations,num_state_discretizations):

	discretized_theta = np.linspace(-np.pi, np.pi, num=num_state_discretizations)
	discretized_theta_dot = np.linspace(-8.0, 8.0, num=num_state_discretizations)
	discretized_action = np.linspace(-2.0,2.0, num=num_action_discretizations)

	return discretized_theta,discretized_theta_dot,discretized_action



def get_next_state_from_eqn(current,action):
	g = 10.
	m = 1.
	l = 1.
	dt = .05
	u = action
	th = current[0]
	thdot = current[1]

	# u = np.clip(, -2., 2.)[0]
	newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
	newth = th + newthdot*dt
	newthdot = np.clip(newthdot, -8., 8.)
	new_state =  np.array([angle_normalize(newth), newthdot])
	return new_state


def find_closest_disc_state(current_state, discretized_theta,discretized_theta_dot):
	disc_theta = discretized_theta[(np.abs(discretized_theta - current_state[0])).argmin()]
	disc_theta_dot = discretized_theta_dot[(np.abs(discretized_theta_dot - current_state[1])).argmin()]
	return (disc_theta,disc_theta_dot)


class PriorityQueue:
    def __init__(self):
        self.elements = []
    
    def empty(self):
        return len(self.elements) == 0
    
    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))
    
    def get(self):
        return heapq.heappop(self.elements)[1]

    def getlength(self):
    	return len(self.elements)    


def get_possible_states(current,discretized_action,discretized_theta,discretized_theta_dot):
	
	children = []
	
	for action in discretized_action:
		new_state = get_next_state_from_eqn(current,action)
		new_state = find_closest_disc_state(new_state,discretized_theta, discretized_theta_dot)
		
		new_state = new_state + (action,)

		children.append(new_state)
		
	return children	


def calculate_h(current, goal):
	# cost = ((goal_node.state[0] - current_node.state[0])**2 + 1*(goal_node.state[1] - current_node.state[1])**2 + 0.001*current_node.parent_action**2)
	#costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)
	action = current[2]
	cost = angle_normalize(current[0])**2 + 0.1*current[1]**2 + 0.001*action**2
	return cost

def reconstruct_path(came_from, start, goal):
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start) # optional
    path.reverse() # optional
    return path


class Node():
    """A node class for A* Pathfinding"""
    def __init__(self, parent_state=None, parent_action=None, state=None):
        self.parent_state = parent_state
        self.parent_action = parent_action 
        self.state = state

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        # return self.state == other.state
        return np.allclose(self.state,other.state, atol=tol)

def calculate_g(current_node, parent_node):
	# cost = (current_node.state[0]**2 + 0.1*current_node.state[1]**2 + 0.001*current_node.parent_action**2)
	# cost = ((parent_node.state[0] - current_node.state[0])**2 + 0.1*(parent_node.state[1] - current_node.state[1])**2 + 0.001*current_node.parent_action**2)

	cost = parent_node.g + 0.001*current_node.parent_action**2
	# cost = 1 + parentcost or parent_action
	return 0



def get_next_state_from_model(model,current_node,action):
	tensor_state_action = torch.tensor(np.array([np.cos(current_node.state[0]),np.sin(current_node.state[0]), current_node.state[1], current_node.state[0],action]).T,dtype = dtype)
	new_state_action = model(tensor_state_action).detach().numpy()
	new_state = np.array([np.arctan2(new_state_action[1],new_state_action[0]),new_state_action[2]]).T
	new_state_disc = find_closest_disc_state(new_state)
	return new_state_disc


def run_sim(action):
	# env.render()
	observation, reward, done, info = env.step([action])
	new_state = np.array([np.arctan2(observation[1],observation[0]),observation[2]]).T
	# print(new_state)
	return new_state 



def astar(model, start, end):
    """Returns a list of actions from the given start to the given end for the given model"""

    # Create start and end node
    start_node = Node(None,0.0, start)
    start_node.g = 0
    start_node.h = calculate_h(start_node,end)
    start_node.f = start_node.g + start_node.h 
  
    end_node = Node(None,0.0, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)



    iterations = 0
    # Loop until you find the end
    while len(open_list) > 0:
        iterations += 1
        # print(iterations)
        print('openlist={},closed_list={}'.format(len(open_list),len(closed_list)))

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        print('currentstate={}, parentaction={}, f={}'.format(current_node.state, current_node.parent_action, current_node.f))

        # run_sim(current_node.parent_action)

        # Found the goal
        if current_node == end_node:    
            actions = []	
            current = current_node
            while current is not None:
                actions.append(current.parent_action)
                current = current.parent_state
                # print(current.state)
            return actions[::-1] # Return reversed path

        # Generate children states
        children = []

        for action in discretized_action: # all possible actions

            # Get new state node (closest to discretized state space)
            
            # new_state = get_next_state_from_model(model,current_node,action)
            new_state = get_next_state_from_eqn(current_node,action)
            # new_state = run_sim(action)
            # print(new_state)

            # # Make sure within range
            # if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
            #     continue

            # # Make sure walkable terrain
            # if maze[node_position[0]][node_position[1]] != 0:
            #     continue

            #############Make sure to wrap theta 
        
            # Create new node
            new_node = Node(current_node, action, new_state)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:	
                    continue

            # Create the f, g, and h values
            child.g = calculate_g(child,current_node) #current_node.g + 1
            child.h = calculate_h(child,end_node)
            child.f = child.g + child.h
            
            # 	print(child.state, child.parent_action, child.f)        
            
            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            open_list.append(child)