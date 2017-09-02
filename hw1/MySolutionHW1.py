#!/usr/bin/env python

# import outside dependencies 
import tensorflow as tf
import numpy as np
import gym
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers import Dense
import pickle

# import self-defined modules
import tf_util
import load_policy

N = 128

def createNN(n_x, n_y):
	model = Sequential()
	
	model.add(Dense(N, activation = 'relu', input_shape = (n_x, ))) 
	model.add(Dense(N, activation = 'relu'))
	model.add(Dense(N, activation = 'relu'))
	model.add(Dense(n_y, activation = 'linear'))
	model.compile(loss = 'msle', optimizer = 'adam', metrics = ['accuracy'])
	return model

def run_the_trained_policy(model, env_name, max_iter, num_running):
	with tf.Session():
		tf_util.initialize() # TODO: try to understand how does initialize work
		env = gym.make(env_name)

		reward_total_list = []
		observs = []
		actions = []
		for _ in range(num_running):
			obs = env.reset()
			done = False
			reward_total = 0
			num_iter = 0
			# while not done:
			while True:
				action = (model.predict(obs[None, :], batch_size = 64)) #TODO: fix batch_size
				# append the last observation caused by last action
				observs.append(obs)
				actions.append(action)
				obs, reward, done, info = env.step(action)
				reward_total += reward
				num_iter += 1
				env.render()
				if num_iter >= max_iter:
					break
			reward_total_list.append(reward_total)
		running_data = {'observations': np.stack(observs, axis = 0), 
		                'actions': np.squeeze(np.stack(actions, axis = 0)),
		                'reward_total_list': np.array(reward_total_list)}
        return running_data

def behavioral_cloning(env_name = 'Hopper-v1', data = 'data/Hopper-v1_20_data.pkl',
					   num_running =3, max_iter = 400):
    with open(data, 'rb') as f:
        training_data = pickle.loads(f.read())

    observ_points = training_data['observations']
    action_points = training_data['actions']

    model = createNN(observ_points.shape[1], action_points.shape[1])
    history = model.fit(observ_points, action_points, batch_size = 64, epochs = 20, verbose = 1)
    running_data = run_the_trained_policy(model, env_name, max_iter, num_running)

behavioral_cloning()
