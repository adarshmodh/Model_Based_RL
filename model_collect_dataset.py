import numpy as np
import gym
import pandas as pd
import csv

env = gym.make('Pendulum-v0')

train_actions = []
train_states = []
train_observations = []

test_actions = []
test_states = []
test_observations = []



for i_episode in range(100):
    print("reset episode")
    observation = env.reset()
    for t in range(1000):
        # env.render()
        # print(observation)
        
        if(i_episode<80):

            action = env.action_space.sample()
            train_actions.append(action)
            
            theta = np.arctan2(observation[1],observation[0])
            traindata = np.append(observation,theta)
            train_states.append(traindata)

            observation, reward, done, info = env.step(action)
            
            train_observations.append(observation)


        else:
            action = env.action_space.sample()
            test_actions.append(action)
            
            theta = np.arctan2(observation[1],observation[0])
            testdata = np.append(observation,theta)
            test_states.append(testdata)
            
            observation, reward, done, info = env.step(action)
            
            test_observations.append(observation)


env.close()

train_data = np.array([train_states, train_actions, train_observations])
print(train_data.shape)

np.save("train_data",train_data)


test_data = np.array([test_states, test_actions, test_observations])
print(test_data.shape)

np.save("test_data",test_data)



