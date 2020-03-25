import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import gym
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

dtype = torch.float
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
D_in, H, D_out = 5, 128, 3

# Create random Tensors to hold input and outputs.
# Setting requires_grad=False indicates that we do not need to compute gradients
# with respect to these Tensors during the backward pass.

test_data = np.vstack(np.load('test_data.npy', allow_pickle=True))

state = np.vstack(test_data[0, :]).astype(np.float)
action = np.vstack(test_data[1, :]).astype(np.float)
observation = np.vstack(test_data[2, :]).astype(np.float)
x_test = np.concatenate((state,action), axis=1)

x = torch.tensor(x_test, requires_grad=False, device=device, dtype = dtype)
y = torch.tensor(observation, requires_grad=False, device=device, dtype = dtype)

print(x.shape,y.shape)

# x = torch.randn(N, D_in, device=device)
# y = torch.randn(N, D_out, device=device)

# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H, bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out, bias=True),
).to(device)


model.load_state_dict(torch.load("pendulum_model.pt", map_location=device))

model.eval()

testloader = DataLoader(TensorDataset(x,y), batch_size = 2000)
# loss_fn = torch.nn.MSELoss(reduction='mean')

hold_loss=[]

def get_next_state_from_eqn(current_state):
	g = 10.
	m = 1.
	l = 1.
	dt = .05
	u = current_state[4]
	th = current_state[3]
	thdot = current_state[2]

	u = np.clip(u, -2., 2.)[0]
	newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
	newth = th + newthdot*dt
	newthdot = np.clip(newthdot, -8., 8.)
	new_state =  np.array([[np.cos(newth), np.sin(newth), newthdot]])
	return new_state
	

for x_batch,y_batch in testloader:
    y_pred = model(x_batch)

    # Compute and print loss.
    
    # print(x_batch.detach().numpy()[0][1])
    # running_loss = mean_squared_error(y_pred.detach().numpy(),get_next_state_from_eqn(x_batch.detach().numpy()[0]))
    running_loss = mean_squared_error(y_pred.detach().numpy(),y_batch.detach().numpy())
    
    print('loss {}'.format(running_loss))

    hold_loss.append(running_loss)

test_error = np.array(hold_loss)
print('test error max= {}, mean = {}' .format(test_error.max(), test_error.mean()))

plt.ylabel('MSE Error')
plt.ylabel('Datapoints')
plt.title('Testing error')

plt.plot(np.array(hold_loss))
plt.show() 

	# env = gym.make('Pendulum-v0')

	# # env = gym.make('Stochastic-4x4-FrozenLake-v0')
	# # p,newstate,reward,terminal = env.P[6][1]
	# # print(p)

	# for i_episode in range(20):
	#     observation = env.reset()
	#     for t in range(100):
	#         env.render()
	#         # print(observation)
	#         action = env.action_space.sample()
	#         observation, reward, done, info = env.step([action])

	#         state = np.append(observation,np.arctan2(observation[1],observation[0]))
	#         tensor_data = torch.tensor(np.append(state,action),dtype = dtype)
	#         predicted_observation = model(tensor_data).detach().numpy().T
	        
	#         print(mean_squared_error(observation.ravel(), predicted_observation))
	#         # with torch.no_grad():
	#         # 	print(loss_fn(observation.ravel(), predicted_observation))
	#         # # print(action)
	        
	#         if done:
	#             print("Episode finished after {} timesteps".format(t+1))
	#             break
	            
	# env.close()