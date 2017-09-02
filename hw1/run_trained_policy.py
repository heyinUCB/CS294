import tensorflow as tf
import numpy as np
import gym
import tf_util
from tensorflow.contrib.keras.python.keras.models import load_model

def run_the_trained_policy(max_iter = 2000, num_running = 3, env_name = 'Hopper-v1'):
	with tf.Session():
		tf_util.initialize() # TODO: try to understand how does initialize work
		env = gym.make(env_name)

		reward_total_list = []
		observs = []
		actions = []
		model = load_model('models/' + env_name + '_cloned_model.h5')
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
        print 'rewards', reward_total_list
        return running_data

run_the_trained_policy()