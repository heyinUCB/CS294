#!/usr/bin/env python

# import outside dependencies 
import tensorflow as tf
import numpy as np
import gym
from tensorflow.contrib.keras.python.keras.models import Sequential, load_model
from tensorflow.contrib.keras.python.keras.layers import Dense
import pickle

# import self-defined modules
import tf_util
import load_policy

N = 128
env_name = 'Hopper-v1'
data = 'data/Hopper-v1_25_data.pkl'

def create_train_NN(observ_points, action_points, n_x, n_y):
	model = Sequential()
	
	model.add(Dense(N, activation = 'sigmoid', input_shape = (n_x, ))) 
	model.add(Dense(N, activation = 'relu'))
	model.add(Dense(n_y, activation = 'linear'))
	model.compile(loss = 'msle', optimizer = 'adam', metrics = ['accuracy'])
	history = model.fit(observ_points, action_points, batch_size = 64, epochs = 40, verbose = 1)
	model.save('models/' + env_name + '_cloned_model.h5')

def behavioral_cloning(num_running =3, max_iter = 1000):
    with open(data, 'rb') as f:
        training_data = pickle.loads(f.read())

    observ_points = training_data['observations']
    action_points = training_data['actions']
    Xy = np.column_stack((observ_points, action_points))
    np.random.shuffle(Xy)

    create_train_NN(Xy[:, :observ_points.shape[1]], Xy[:, observ_points.shape[1]:], observ_points.shape[1], action_points.shape[1])

behavioral_cloning()
