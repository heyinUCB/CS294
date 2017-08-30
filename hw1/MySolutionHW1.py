#!/usr/bin/env python

# import outside dependencies 
import tensorflow as tf
import numpy as np
import gym
from tensorflow.contrib import keras
from keras.models import Sequential
from keras.layers import Dense

# import self-defined modules
import tf_util
import load_policy

# Createa CNN with the architecture [INPUT - CONV - RELU - POOL - FC]
def cnn(batch_size = 100):
	model = Sequential()
	
	model.add(Dense(batch_size, activation = 'conv', input_shape = (n_x, ))) 
	# TODO: fix the input of activation, figure out the input shape
	model.add(Dense(batch_size, activation = 'relu'))
	model.add(Dense(batch_size, activation = 'pool'))
	# TODO: figure out input of activation
	model.add(Dense(n_y, activation = 'FC'))
	# TODO: figure out n_y, and input of activation
	model.compile(loss = 'msle', optimizer = 'adam', metrics = ['accuracy'])
    # TODO: figure out the inputs of compile...
    return model

def get_training_data(num_running, stop_iter, policy_fn, env):
	with tf.Session():
		tf_util.initialize() # TODO: try to understand how does initialize work

		reward_total_list = []
		observs = []
		actions = []
		for _ in range(num_running):
			obs = env.reset()
			done = False
			reward_total = 0
			num_iter = 0
			while not done:
				action = policy_fn(obs[None, :])
				# append the last observation caused by last action
				observs.append(obs)
				actions.append(action)
				obs, reward, done, info = env.step(action)
				reward_total += reward
				num_iter += 1
				# env.render()
				if num_iter >= stop_iter:
					break
			reward_total_list.append(reward_total)
		train_data_expert = {'observations': np.stack(observs, axis = 0), 
		                     'actions': np.squeeze(np.stack(actions, axis = 0)),
		                     'reward_total_list': np.array(reward_total_list)}
        return train_data_expert

def Behavioral_cloning(num_rollouts = 2, stop_iter = 2000, env_name = 'Hopper-v1', \
					   expert_policy_file = 'experts/Hopper-v1.pkl', num_running = 50):
	tf.reset_default_graph()
	expert_policy = load_policy.load_policy(expert_policy_file)
	env = gym.make(env_name)
	training_data = get_training_data(num_rollouts, stop_iter, expert_policy, env)
    
    # with tf.Session():
    # 	tf_util.initialize()

    # 	for epoch in range(num_running):

Behavioral_cloning()






